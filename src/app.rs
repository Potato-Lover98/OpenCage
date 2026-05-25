use std::collections::VecDeque;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::time::SystemTime;

use anyhow::{Context, Result};
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyboardEnhancementFlags, KeyCode, KeyEvent,
    KeyEventKind, KeyModifiers, MouseButton, MouseEvent, MouseEventKind, PopKeyboardEnhancementFlags,
    PushKeyboardEnhancementFlags,
};
use ratatui::layout::{Position, Rect};
use crossterm::execute;
use crossterm::terminal::{
    size,
    EnterAlternateScreen, LeaveAlternateScreen, SetTitle, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use serde::Deserialize;
use walkdir::WalkDir;

use crate::ai::image_paste::{encode_as_data_url, grab_clipboard_image};
use crate::ai::providers::{
    provider_has_credentials, query_coding_actions, query_with_subagent, select_subagents,
    validate_settings_keys,
};
use crate::ai::sandbox::run_in_sandbox;
use crate::ai::rag::RagStore;
use crate::ai::telegram;
use crate::ai::voice::{
    render_level_bar, spawn_live_stt, CloudSttConfig, VoiceRecorder, VoiceSttUpdate,
};
use crate::core::config::SettingsStore;
use crate::core::migration::{self, MigrationSource};
use crate::core::models::{FileNode, Message, Provider, Settings};
use crate::core::plugins;
use crate::ui;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveTab {
    Chat,
    Settings,
    Alerts,
}

#[derive(Debug, Clone)]
pub enum AlertKind {
    Info,
    Success,
    Warning,
    Error,
}

#[derive(Debug, Clone)]
pub struct AlertItem {
    pub at: String,
    pub kind: AlertKind,
    pub title: String,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct SettingsForm {
    pub selected: usize,
    pub provider_idx: usize,
    pub model_idx: usize,
    pub avatar: String,
    pub cuss_filter: bool,
    pub openai: String,
    pub groq: String,
    pub anthropic: String,
    pub moonshot: String,
    pub glm: String,
    pub copilot: String,
    /// Selected source (index into `MigrationSource::all()`) for the migration row.
    pub migration_idx: usize,
    pub report: Vec<String>,
}

pub struct App {
    pub settings: Settings,
    pub input: String,
    pub messages: Vec<Message>,
    pub files: Vec<FileNode>,
    pub selected_file_idx: usize,
    /// When false, the file tree list shows no cyan selection (e.g. after clicking the chat area).
    pub file_tree_highlight: bool,
    pub show_palette: bool,
    pub palette_commands: Vec<&'static str>,
    pub palette_selected: usize,
    pub chat_scroll: usize,
    pub busy: bool,
    pub pending_response: Option<Receiver<AgentResult>>,
    pub buddy_mode: bool,
    pub last_status: String,
    pub deep_think_enabled: bool,
    pub deep_think_level: u8,
    pub coding_expanded: bool,
    pub active_tab: ActiveTab,
    pub settings_form: Option<SettingsForm>,
    pub current_dir: PathBuf,
    pub current_dir_trusted: bool,
    last_tree_refresh: Instant,
    /// When false, only the project root row is shown (children hidden until expanded again).
    file_tree_root_expanded: bool,
    expanded_paths: HashSet<PathBuf>,
    pub context_window_limit_chars: usize,
    pending_command: Option<String>,
    command_approval_yes: bool,
    queued_commands: VecDeque<String>,
    pending_trust_prompt: bool,
    deferred_coding_prompt: Option<String>,
    rag: RagStore,
    blacklist_last_modified: Option<SystemTime>,
    voice_recorder: Option<VoiceRecorder>,
    voice_level: Arc<AtomicU8>,
    voice_transcribing: bool,
    voice_rx: Option<Receiver<VoiceSttUpdate>>,
    pub voice_partial: String,
    voice_live_active: Option<Arc<AtomicBool>>,
    attached_image_path: Option<PathBuf>,
    /// F9 held for push-to-talk: first press time; hold ≥1s → voice (Space is never voice).
    voice_f9_press_start: Option<Instant>,
    session_id: String,
    last_persisted_messages_len: usize,
    show_resume_hint_on_exit: bool,
    show_sessions_popup: bool,
    sessions_menu: Vec<SessionMeta>,
    sessions_selected: usize,
    alerts: VecDeque<AlertItem>,
    /// Last terminal tab title we wrote, so we only re-emit the OSC sequence on change.
    tab_title: String,
    /// Set the flag to stop the Telegram polling thread; `Some` means a bot is running.
    telegram_stop: Option<Arc<AtomicBool>>,
    /// Activity lines from the Telegram thread, drained into the chat as system messages.
    telegram_rx: Option<Receiver<String>>,
    /// Set the flag to stop the Beetle Design web server; `Some` means it's running.
    beetle_stop: Option<Arc<AtomicBool>>,
    /// URL the Beetle Design server is listening on, for reopening the browser.
    beetle_url: Option<String>,
    show_provider_popup: bool,
    provider_menu: Vec<Provider>,
    provider_selected: usize,
}

pub struct AgentResult {
    pub text: String,
    pub commands: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct SessionFile {
    title: String,
    updated_ts: u64,
    messages: Vec<Message>,
}

#[derive(Debug, Clone)]
pub struct SessionMeta {
    pub id: String,
    pub title: String,
    pub updated_ts: u64,
}

impl App {
    pub fn new(settings: Settings, rag: RagStore, resume_session: Option<String>) -> Self {
        let current_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let current_dir_key = current_dir.to_string_lossy().to_string();
        let trusted = settings.trusted_paths.contains(&current_dir_key);
        let session_id = resume_session.unwrap_or_else(generate_session_id);
        let mut app = Self {
            settings,
            input: String::new(),
            messages: vec![Message {
                role: "system".to_string(),
                content: "Welcome to Opencage. Type / to open command palette.".to_string(),
            }],
            files: vec![],
            selected_file_idx: 0,
            file_tree_highlight: true,
            show_palette: false,
            palette_commands: vec![
                "/alerts",
                "/settings",
                "/sessions",
                "/new",
                "/help",
                "/buddy",
                "/btw",
                "/model",
                "/blacklist",
                "/avatar",
                "/clear",
                "/refresh",
                "/remember",
                "/memories",
                "/deep",
                "/expand",
                "/voice",
                "/migration",
                "/anthropic",
                "/plugins",
                "/plugout",
                "/bots",
                "/beetle",
            ],
            palette_selected: 0,
            chat_scroll: 0,
            busy: false,
            pending_response: None,
            buddy_mode: false,
            last_status: "Ready".to_string(),
            deep_think_enabled: false,
            deep_think_level: 2,
            coding_expanded: false,
            active_tab: ActiveTab::Chat,
            settings_form: None,
            current_dir,
            current_dir_trusted: trusted,
            last_tree_refresh: Instant::now(),
            file_tree_root_expanded: true,
            expanded_paths: HashSet::new(),
            context_window_limit_chars: crate::ai::providers::CONTEXT_BUDGET_CHARS,
            pending_command: None,
            command_approval_yes: true,
            queued_commands: VecDeque::new(),
            pending_trust_prompt: false,
            deferred_coding_prompt: None,
            rag,
            blacklist_last_modified: None,
            voice_recorder: None,
            voice_level: Arc::new(AtomicU8::new(0)),
            voice_transcribing: false,
            voice_rx: None,
            voice_partial: String::new(),
            voice_live_active: None,
            attached_image_path: None,
            voice_f9_press_start: None,
            session_id,
            last_persisted_messages_len: 0,
            show_resume_hint_on_exit: false,
            show_sessions_popup: false,
            sessions_menu: vec![],
            sessions_selected: 0,
            alerts: VecDeque::new(),
            tab_title: String::new(),
            telegram_stop: None,
            telegram_rx: None,
            beetle_stop: None,
            beetle_url: None,
            show_provider_popup: false,
            provider_menu: Vec::new(),
            provider_selected: 0,
        };
        if let Err(e) = app.load_session_messages() {
            app.messages.push(Message {
                role: "system".to_string(),
                content: format!("Session load warning: {e}"),
            });
        }
        if app.messages.is_empty() {
            app.push_system_message("Welcome to Opencage. Type / to open command palette.");
        }
        if !trusted {
            app.push_alert(
                AlertKind::Warning,
                "Untrusted folder",
                "This folder is not trusted yet. Coding tasks will ask yes/no first.",
            );
            app.push_system_message(
                "This folder is not trusted yet. When coding is requested, I will ask yes/no before trusting it.",
            );
        }
        app.refresh_file_tree();
        app.sync_blacklist_file();
        app
    }

    pub fn run(&mut self) -> Result<()> {
        enable_raw_mode()?;
        let mut stdout = std::io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        // Lets Shift+Enter / Ctrl+Enter report modifiers in terminals that support CSI-u (Kitty, WezTerm, Alacritty, foot).
        let _ = execute!(
            stdout,
            PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
        );
        let backend = ratatui::backend::CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;
        let tick = Duration::from_millis(120);
        let mut should_quit = false;

        // Resume the Telegram bridge automatically if a token was saved.
        if self
            .settings
            .telegram_bot_token
            .as_ref()
            .is_some_and(|t| !t.trim().is_empty())
        {
            self.start_telegram_bot();
        }

        while !should_quit {
            self.sync_cwd_and_tree();
            self.sync_blacklist_file();
            self.sync_voice_state();
            self.persist_session_if_needed();
            self.sync_terminal_title();
            self.drain_telegram_log();
            self.sync_beetle_state();
            terminal.draw(|f| ui::draw(f, self))?;

            if let Some(rx) = self.pending_response.as_ref() {
                match rx.try_recv() {
                    Ok(result) => {
                        let final_reply = self.filter_cuss(result.text);
                        self.messages.push(Message {
                            role: "assistant".to_string(),
                            content: final_reply.clone(),
                        });
                        for cmd in result.commands {
                            self.queued_commands.push_back(cmd);
                        }
                        self.prompt_next_command_if_needed();
                        let _ = self.rag.remember_scoped(
                            &format!("session:{}:global", self.session_id),
                            "assistant",
                            &final_reply,
                        );
                        for line in final_reply.lines() {
                            if let Some(agent_name) = parse_agent_tag(line) {
                                let scope = format!("session:{}:agent:{agent_name}", self.session_id);
                                let _ = self.rag.remember_scoped(&scope, "assistant", line);
                            }
                        }
                        self.busy = false;
                        self.pending_response = None;
                        self.last_status = "Ready".to_string();
                        self.chat_scroll = 0;
                    }
                    Err(TryRecvError::Disconnected) => {
                        self.busy = false;
                        self.pending_response = None;
                        self.last_status = "Provider task ended unexpectedly".to_string();
                    }
                    Err(TryRecvError::Empty) => {}
                }
            }

            if event::poll(tick)? {
                match event::read()? {
                    Event::Key(key) => should_quit = self.handle_key(key),
                    Event::Mouse(mouse) => self.handle_mouse(mouse),
                    _ => {}
                }
            }
        }

        if let Some(stop) = self.telegram_stop.take() {
            stop.store(true, Ordering::SeqCst);
        }
        if let Some(stop) = self.beetle_stop.take() {
            stop.store(true, Ordering::SeqCst);
        }
        disable_raw_mode()?;
        let _ = execute!(terminal.backend_mut(), PopKeyboardEnhancementFlags);
        let _ = execute!(terminal.backend_mut(), SetTitle(""));
        execute!(terminal.backend_mut(), DisableMouseCapture, LeaveAlternateScreen)?;
        terminal.show_cursor()?;
        if self.show_resume_hint_on_exit {
            eprintln!("Resume this session with: opencage resume {}", self.session_id);
        }
        Ok(())
    }

    fn voice_ptt_arm_allowed(&self) -> bool {
        self.active_tab == ActiveTab::Chat
            && !self.show_palette
            && !self.busy
            && self.pending_command.is_none()
            && !self.pending_trust_prompt
    }

    /// After F9 is held ~1s without starting voice yet, begin recording.
    fn try_start_voice_after_f9_hold(&mut self) {
        if self.voice_recorder.is_some() || self.voice_transcribing {
            return;
        }
        if !self.voice_ptt_arm_allowed() {
            self.voice_f9_press_start = None;
            return;
        }
        let Some(t0) = self.voice_f9_press_start else {
            return;
        };
        if t0.elapsed() >= Duration::from_secs(1) {
            self.voice_f9_press_start = None;
            self.toggle_voice_recording();
        }
    }

    fn handle_key(&mut self, key: KeyEvent) -> bool {
        if key.code == KeyCode::F(9)
            && key.modifiers.is_empty()
            && self.voice_ptt_arm_allowed()
        {
            match key.kind {
                KeyEventKind::Press => {
                    if self.voice_recorder.is_none() && !self.voice_transcribing {
                        if self.voice_f9_press_start.is_none() {
                            self.voice_f9_press_start = Some(Instant::now());
                        }
                    }
                    return false;
                }
                KeyEventKind::Release => {
                    if self.voice_recorder.is_none() && !self.voice_transcribing {
                        self.voice_f9_press_start.take();
                    } else {
                        self.voice_f9_press_start = None;
                    }
                    return false;
                }
                _ => {}
            }
        }

        if self.active_tab == ActiveTab::Settings {
            return self.handle_settings_key(key);
        }
        if self.active_tab == ActiveTab::Alerts {
            match key.code {
                KeyCode::Esc => {
                    self.active_tab = ActiveTab::Chat;
                    self.last_status = "Closed alerts tab".to_string();
                    return false;
                }
                KeyCode::Char('c') if key.modifiers == KeyModifiers::CONTROL => {
                    self.alerts.clear();
                    self.last_status = "Alerts cleared".to_string();
                    return false;
                }
                _ => {}
            }
        }
        if self.show_sessions_popup {
            match key.code {
                KeyCode::Esc => {
                    self.show_sessions_popup = false;
                    self.last_status = "Closed sessions menu".to_string();
                    return false;
                }
                KeyCode::Up => {
                    self.sessions_selected = self.sessions_selected.saturating_sub(1);
                    return false;
                }
                KeyCode::Down => {
                    if !self.sessions_menu.is_empty() {
                        self.sessions_selected =
                            (self.sessions_selected + 1).min(self.sessions_menu.len() - 1);
                    }
                    return false;
                }
                KeyCode::Enter
                    if !key.modifiers.contains(KeyModifiers::SHIFT)
                        && !key.modifiers.contains(KeyModifiers::ALT) =>
                {
                    self.switch_to_selected_session();
                    return false;
                }
                _ => return false,
            }
        }
        if self.show_provider_popup {
            match key.code {
                KeyCode::Esc => {
                    self.show_provider_popup = false;
                    self.last_status = "Closed provider switch".to_string();
                    return false;
                }
                KeyCode::Up => {
                    self.provider_selected = self.provider_selected.saturating_sub(1);
                    return false;
                }
                KeyCode::Down => {
                    if !self.provider_menu.is_empty() {
                        self.provider_selected =
                            (self.provider_selected + 1).min(self.provider_menu.len() - 1);
                    }
                    return false;
                }
                KeyCode::Enter => {
                    self.apply_provider_switch();
                    return false;
                }
                _ => return false,
            }
        }
        if self.pending_command.is_some() {
            match key.code {
                KeyCode::Left => {
                    self.command_approval_yes = true;
                    return false;
                }
                KeyCode::Right => {
                    self.command_approval_yes = false;
                    return false;
                }
                KeyCode::Enter
                    if !key.modifiers.contains(KeyModifiers::SHIFT)
                        && !key.modifiers.contains(KeyModifiers::ALT) =>
                {
                    self.apply_pending_command_choice(self.command_approval_yes);
                    return false;
                }
                _ => {}
            }
        }
        if self.show_palette {
            let suggestions = self.filtered_suggestions();
            match key.code {
                KeyCode::Up => {
                    self.palette_selected = self.palette_selected.saturating_sub(1);
                    return false;
                }
                KeyCode::Down => {
                    if !suggestions.is_empty() {
                        self.palette_selected = (self.palette_selected + 1).min(suggestions.len() - 1);
                    }
                    return false;
                }
                KeyCode::Enter
                    if !key.modifiers.contains(KeyModifiers::SHIFT)
                        && !key.modifiers.contains(KeyModifiers::ALT) =>
                {
                    if let Some(selected) = suggestions.get(self.palette_selected) {
                        self.input = (*selected).to_string();
                        self.show_palette = false;
                        self.palette_selected = 0;
                        return false;
                    }
                }
                KeyCode::Esc => {
                    self.show_palette = false;
                    self.palette_selected = 0;
                    return false;
                }
                KeyCode::PageUp => {
                    self.palette_selected = self.palette_selected.saturating_sub(5);
                    return false;
                }
                KeyCode::PageDown => {
                    if !suggestions.is_empty() {
                        self.palette_selected =
                            (self.palette_selected + 5).min(suggestions.len() - 1);
                    }
                    return false;
                }
                _ => {}
            }
        }

        match key {
            KeyEvent {
                code: KeyCode::Char('v'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL)
                && modifiers.contains(KeyModifiers::SHIFT) =>
            {
                if self.active_tab == ActiveTab::Chat {
                    match grab_clipboard_image() {
                        Ok(img) => {
                            self.attached_image_path = Some(img.path.clone());
                            let kb = (img.size_bytes.saturating_add(1023)) / 1024;
                            self.push_system_message(&format!(
                                "image pasted {}kb (will be sent with next message)",
                                kb
                            ));
                            self.last_status = format!("Image pasted: {kb}kb");
                        }
                        Err(e) => {
                            self.push_alert(
                                AlertKind::Warning,
                                "Clipboard paste failed",
                                &e.to_string(),
                            );
                            self.last_status = format!("Paste failed: {e}");
                        }
                    }
                }
                return false;
            }
            KeyEvent {
                code: KeyCode::Char('c'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                self.show_resume_hint_on_exit = true;
                return true;
            }
            KeyEvent {
                code: KeyCode::Tab, ..
            } => {
                self.file_tree_highlight = true;
                self.toggle_node();
            }
            KeyEvent {
                code: KeyCode::Left,
                modifiers: KeyModifiers::ALT,
                ..
            } => {
                if self.deep_think_enabled {
                    self.deep_think_level = self.deep_think_level.saturating_sub(1);
                    self.last_status = format!(
                        "Deep Think level {} (higher uses more tokens)",
                        self.deep_think_level
                    );
                } else {
                    self.last_status = "Deep Think is OFF. Use /deep on first.".to_string();
                }
            }
            KeyEvent {
                code: KeyCode::Right,
                modifiers: KeyModifiers::ALT,
                ..
            } => {
                if self.deep_think_enabled {
                    self.deep_think_level = (self.deep_think_level + 1).min(10);
                    self.last_status = format!(
                        "Deep Think level {} (higher uses more tokens)",
                        self.deep_think_level
                    );
                } else {
                    self.last_status = "Deep Think is OFF. Use /deep on first.".to_string();
                }
            }
            KeyEvent {
                code: KeyCode::Char('o'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                self.coding_expanded = !self.coding_expanded;
                self.last_status = format!(
                    "Coding mode: {}",
                    if self.coding_expanded {
                        "expanded (detailed output)"
                    } else {
                        "compact (short code)"
                    }
                );
            }
            KeyEvent {
                code: KeyCode::Char('v'),
                modifiers: KeyModifiers::ALT,
                ..
            } => {
                self.toggle_voice_recording();
            }
            KeyEvent {
                code: KeyCode::PageUp,
                ..
            } => {
                self.chat_scroll = self.chat_scroll.saturating_add(8);
            }
            KeyEvent {
                code: KeyCode::PageDown,
                ..
            } => {
                self.chat_scroll = self.chat_scroll.saturating_sub(8);
            }
            KeyEvent {
                code: KeyCode::Up,
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => {
                self.chat_scroll = self.chat_scroll.saturating_add(3);
            }
            KeyEvent {
                code: KeyCode::Down,
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => {
                self.chat_scroll = self.chat_scroll.saturating_sub(3);
            }
            KeyEvent {
                code: KeyCode::Up, ..
            } => {
                self.file_tree_highlight = true;
                self.selected_file_idx = self.selected_file_idx.saturating_sub(1);
            }
            KeyEvent {
                code: KeyCode::Down,
                ..
            } => {
                self.file_tree_highlight = true;
                self.selected_file_idx = (self.selected_file_idx + 1).min(self.files.len() - 1);
            }
            KeyEvent {
                code: KeyCode::Char('j' | 'J'),
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::CONTROL) => {
                self.input.push('\n');
                self.show_palette = self.input.starts_with('/');
                return false;
            }
            KeyEvent {
                code: KeyCode::Enter,
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::SHIFT) || modifiers.contains(KeyModifiers::ALT) => {
                self.input.push('\n');
                self.show_palette = self.input.starts_with('/');
                return false;
            }
            KeyEvent {
                code: KeyCode::Enter,
                ..
            } => self.handle_enter(),
            KeyEvent {
                code: KeyCode::Backspace,
                ..
            } => {
                self.input.pop();
                self.show_palette = self.input.starts_with('/');
                if !self.show_palette {
                    self.palette_selected = 0;
                }
            }
            KeyEvent {
                code: KeyCode::Char(ch),
                ..
            } => {
                self.input.push(ch);
                self.show_palette = self.input.starts_with('/');
                if !self.show_palette {
                    self.palette_selected = 0;
                }
            }
            _ => {}
        }
        false
    }

    fn handle_settings_key(&mut self, key: KeyEvent) -> bool {
        let Some(form) = self.settings_form.as_mut() else {
            self.active_tab = ActiveTab::Chat;
            return false;
        };
        match key {
            KeyEvent {
                code: KeyCode::Esc, ..
            } => {
                self.active_tab = ActiveTab::Chat;
                self.last_status = "Closed settings tab".to_string();
            }
            KeyEvent {
                code: KeyCode::Up, ..
            } => {
                form.selected = form.selected.saturating_sub(1);
            }
            KeyEvent {
                code: KeyCode::Down,
                ..
            } => {
                form.selected = (form.selected + 1).min(10);
            }
            KeyEvent {
                code: KeyCode::Left,
                ..
            } if form.selected == 0 => {
                let total = Provider::all().len();
                form.provider_idx = (form.provider_idx + total - 1) % total;
                let provider = &Provider::all()[form.provider_idx];
                let models = provider.models();
                if form.model_idx >= models.len() {
                    form.model_idx = 0;
                }
            }
            KeyEvent {
                code: KeyCode::Right,
                ..
            } if form.selected == 0 => {
                let total = Provider::all().len();
                form.provider_idx = (form.provider_idx + 1) % total;
                let provider = &Provider::all()[form.provider_idx];
                let models = provider.models();
                if form.model_idx >= models.len() {
                    form.model_idx = 0;
                }
            }
            KeyEvent {
                code: KeyCode::Left,
                ..
            } if form.selected == 1 => {
                let provider = &Provider::all()[form.provider_idx];
                let total = provider.models().len();
                form.model_idx = (form.model_idx + total - 1) % total;
            }
            KeyEvent {
                code: KeyCode::Right,
                ..
            } if form.selected == 1 => {
                let provider = &Provider::all()[form.provider_idx];
                let total = provider.models().len();
                form.model_idx = (form.model_idx + 1) % total;
            }
            KeyEvent {
                code: KeyCode::Left,
                ..
            } if form.selected == 10 => {
                let total = MigrationSource::all().len();
                form.migration_idx = (form.migration_idx + total - 1) % total;
            }
            KeyEvent {
                code: KeyCode::Right,
                ..
            } if form.selected == 10 => {
                let total = MigrationSource::all().len();
                form.migration_idx = (form.migration_idx + 1) % total;
            }
            KeyEvent {
                code: KeyCode::Enter,
                ..
            } if form.selected == 3 => {
                form.cuss_filter = !form.cuss_filter;
            }
            KeyEvent {
                code: KeyCode::Enter,
                ..
            } if form.selected == 10 => {
                let source = MigrationSource::all()[form.migration_idx];
                self.run_migration_source(source);
            }
            KeyEvent {
                code: KeyCode::Char('s'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => self.save_settings_form(),
            KeyEvent {
                code: KeyCode::F(5),
                ..
            } => self.validate_settings_form(),
            KeyEvent {
                code: KeyCode::Backspace,
                ..
            } => {
                if matches!(form.selected, 2 | 4 | 5 | 6 | 7 | 8 | 9) {
                    Self::form_field_mut(form).pop();
                }
            }
            KeyEvent {
                code: KeyCode::Char(ch),
                ..
            } => {
                if matches!(form.selected, 2 | 4 | 5 | 6 | 7 | 8 | 9) {
                    Self::form_field_mut(form).push(ch);
                }
            }
            _ => {}
        }
        false
    }

    fn form_field_mut(form: &mut SettingsForm) -> &mut String {
        match form.selected {
            2 => &mut form.avatar,
            4 => &mut form.openai,
            5 => &mut form.groq,
            6 => &mut form.anthropic,
            7 => &mut form.moonshot,
            8 => &mut form.glm,
            9 => &mut form.copilot,
            _ => &mut form.avatar,
        }
    }

    fn handle_mouse(&mut self, mouse: MouseEvent) {
        match mouse.kind {
            MouseEventKind::ScrollUp | MouseEventKind::ScrollDown => {
                self.handle_chat_mouse_scroll(mouse);
                return;
            }
            MouseEventKind::Down(MouseButton::Left) => {}
            _ => return,
        }
        if self.show_sessions_popup {
            self.try_sessions_delete_click(mouse);
            return;
        }
        if self.show_provider_popup {
            self.try_provider_click(mouse);
            return;
        }
        if self.active_tab == ActiveTab::Alerts && self.try_alerts_command_click(mouse) {
            return;
        }
        if self.show_palette && let Some((start, end)) = self.suggestion_row_range() {
            let row = mouse.row as usize;
            if row >= start && row < end {
                let idx = row - start;
                let suggestions = self.filtered_suggestions();
                if let Some(selected) = suggestions.get(idx) {
                    self.input = (*selected).to_string();
                    self.show_palette = false;
                    self.palette_selected = 0;
                    return;
                }
            }
        }
        if self.active_tab != ActiveTab::Chat {
            return;
        }
        let Some((w, h)) = size().ok() else {
            return;
        };
        let term = Rect::new(0, 0, w, h);
        let layout = ui::chat_main_layout(
            term,
            self.show_palette,
            self.command_approval_bar().is_some(),
        );
        let pos = Position::new(mouse.column, mouse.row);
        let tree_inner = ui::file_tree_content_rect(layout.files, &self.current_dir);
        if tree_inner.contains(pos) {
            let row = mouse.row.saturating_sub(tree_inner.y) as usize;
            if row < self.files.len() {
                self.file_tree_highlight = true;
                self.selected_file_idx = row;
                if self.files[row].is_dir {
                    self.toggle_node();
                }
            }
        } else if layout.chat.contains(pos)
            || layout.footer.contains(pos)
            || (layout.palette.height > 0 && layout.palette.contains(pos))
        {
            self.file_tree_highlight = false;
        }
    }

    /// Wheel over chat: scroll history (like WhatsApp). Up = older, down = newer.
    fn handle_chat_mouse_scroll(&mut self, mouse: MouseEvent) {
        if self.active_tab != ActiveTab::Chat {
            return;
        }
        let Some((w, h)) = size().ok() else {
            return;
        };
        let term = Rect::new(0, 0, w, h);
        let layout = ui::chat_main_layout(
            term,
            self.show_palette,
            self.command_approval_bar().is_some(),
        );
        let inner = ui::chat_message_inner_rect(layout.chat, self);
        let pos = Position::new(mouse.column, mouse.row);
        if !inner.contains(pos) {
            return;
        }
        const LINES: usize = 3;
        match mouse.kind {
            MouseEventKind::ScrollUp => {
                self.chat_scroll = self.chat_scroll.saturating_add(LINES);
            }
            MouseEventKind::ScrollDown => {
                self.chat_scroll = self.chat_scroll.saturating_sub(LINES);
            }
            _ => {}
        }
    }

    fn handle_enter(&mut self) {
        let raw = self.input.trim().to_string();
        if raw.is_empty() || self.busy {
            return;
        }
        self.input.clear();
        if self.pending_trust_prompt {
            match parse_yes_no(&raw) {
                Some(true) => {
                    let key = self.current_dir.to_string_lossy().to_string();
                    self.settings.trusted_paths.insert(key);
                    self.current_dir_trusted = true;
                    let _ = SettingsStore::save(&self.settings);
                    self.pending_trust_prompt = false;
                    self.messages.push(Message {
                        role: "system".to_string(),
                        content: format!("Trusted: {}", self.current_dir.display()),
                    });
                    if let Some(prompt) = self.deferred_coding_prompt.take() {
                        self.start_autocode(prompt);
                    }
                }
                Some(false) => {
                    self.pending_trust_prompt = false;
                    self.deferred_coding_prompt = None;
                    self.messages.push(Message {
                        role: "system".to_string(),
                        content: "Trust denied. Autonomous coding cancelled.".to_string(),
                    });
                }
                None => {
                    self.messages.push(Message {
                        role: "system".to_string(),
                        content: "Please answer yes/y or no/n.".to_string(),
                    });
                }
            }
            return;
        }
        if self.pending_command.is_some() {
            match parse_yes_no(&raw) {
                Some(true) => self.apply_pending_command_choice(true),
                Some(false) => self.apply_pending_command_choice(false),
                None => {
                    self.messages.push(Message {
                        role: "system".to_string(),
                        content: "Please answer yes/y or no/n for command approval.".to_string(),
                    });
                }
            }
            return;
        }
        if raw.starts_with('/') {
            self.execute_command(raw);
            return;
        }
        if let Some(rest) = raw.strip_prefix("!run ") {
            self.queued_commands.push_back(rest.trim().to_string());
            self.prompt_next_command_if_needed();
            self.last_status = "Command queued for approval".to_string();
            return;
        }

        let user_text = self.filter_cuss(raw);
        self.messages.push(Message {
            role: "user".to_string(),
            content: user_text.clone(),
        });
        self.chat_scroll = 0;
        let _ = self.rag.remember_scoped(
            &format!("session:{}:global", self.session_id),
            "user",
            &user_text,
        );

        if is_coding_task(&user_text) {
            if !self.current_dir_trusted {
                self.pending_trust_prompt = true;
                self.deferred_coding_prompt = Some(user_text.clone());
                self.messages.push(Message {
                    role: "system".to_string(),
                    content: format!(
                        "Trust this folder for autonomous coding?\n{}\nType yes/y or no/n.",
                        self.current_dir.display()
                    ),
                });
                return;
            }
            self.start_autocode(user_text.clone());
            return;
        }

        self.busy = true;
        self.last_status = "Contacting cloud provider...".to_string();
        let (tx, rx) = mpsc::channel();
        let settings = self.settings.clone();
        let context_messages = self.messages.clone();
        let buddy_mode = self.buddy_mode;
        let deep_think_enabled = self.deep_think_enabled;
        let deep_think_level = self.deep_think_level;
        let coding_expanded = self.coding_expanded;
        let selected_agents = select_subagents(&user_text);
        let mut rag_per_agent: Vec<(String, Vec<String>)> = vec![];
        for agent in &selected_agents {
            let scope = format!("session:{}:agent:{}", self.session_id, agent.name());
            let _ = self.rag.remember_scoped(&scope, "user", &user_text);
            let ctx = self
                .rag
                .retrieve_scoped(&scope, &user_text, 3)
                .unwrap_or_default();
            rag_per_agent.push((scope, ctx));
        }
        let image_data_url = self.attached_image_path.take().and_then(|p| {
            match encode_as_data_url(&p) {
                Ok(url) => Some(url),
                Err(_) => None,
            }
        });
        std::thread::spawn(move || {
            let mut replies = Vec::new();
            for (idx, agent) in selected_agents.iter().enumerate() {
                let rag_context = rag_per_agent
                    .get(idx)
                    .map(|(_, c)| c.clone())
                    .unwrap_or_default();
                let response = query_with_subagent(
                    &settings,
                    &context_messages,
                    &user_text,
                    buddy_mode,
                    true,
                    deep_think_enabled,
                    deep_think_level,
                    coding_expanded,
                    *agent,
                    &rag_context,
                    image_data_url.as_deref(),
                )
                .unwrap_or_else(|e| format!("Provider error [{}]: {e}", agent.name()));
                replies.push(response);
            }
            let reply = replies.join("\n\n");
            let _ = tx.send(AgentResult {
                text: reply,
                commands: vec![],
            });
        });
        self.pending_response = Some(rx);
    }

    fn execute_command(&mut self, cmd: String) {
        let mut parts = cmd.split_whitespace();
        let name = parts.next().unwrap_or_default();
        match name {
            "/alerts" => {
                self.active_tab = ActiveTab::Alerts;
                self.last_status = "Alerts tab opened".to_string();
            }
            "/settings" => {
                self.open_settings_tab();
            }
            "/help" => {
                self.push_system_message(
                    "Commands: /alerts /settings /avatar /model[ switch] /buddy /btw /new /sessions /blacklist /clear /refresh /remember /memories /deep(on|off|0-10) /expand(on|off) /migration(claude|openclaw|hermes) /anthropic models <opus-4.7|sonnet-4.7|haiku-4.7> /plugins [available] /plugout <name> /bots <token> /beetle",
                );
            }
            "/buddy" => {
                self.buddy_mode = !self.buddy_mode;
                self.messages.push(Message {
                    role: "system".to_string(),
                    content: format!(
                        "Buddy mode {}",
                        if self.buddy_mode { "enabled" } else { "disabled" }
                    ),
                });
            }
            "/btw" => {
                let side_q = parts.collect::<Vec<_>>().join(" ");
                if side_q.is_empty() {
                    self.messages.push(Message {
                        role: "system".to_string(),
                        content: "Usage: /btw <side question>".to_string(),
                    });
                } else {
                    self.input = format!("BTW: {side_q}");
                    self.handle_enter();
                }
            }
            "/model" => match parts.next().unwrap_or_default() {
                "switch" => self.open_provider_switch(),
                "" => self.messages.push(Message {
                    role: "system".to_string(),
                    content: format!(
                        "Current provider={} model={}",
                        self.settings.provider.as_str(),
                        self.settings.model
                    ),
                }),
                _ => self.push_system_message("Usage: /model [switch]"),
            },
            "/avatar" => {
                let avatar = parts.collect::<Vec<_>>().join(" ");
                if avatar.is_empty() {
                    self.messages.push(Message {
                        role: "system".to_string(),
                        content: format!("Current avatar: {}", self.settings.ai_avatar),
                    });
                } else {
                    self.settings.ai_avatar = avatar;
                    if let Err(e) = SettingsStore::save(&self.settings) {
                        self.last_status = format!("Settings save failed: {e}");
                    }
                    self.messages.push(Message {
                        role: "system".to_string(),
                        content: format!("Avatar updated to {}", self.settings.ai_avatar),
                    });
                }
            }
            "/blacklist" => {
                let args = parts.collect::<Vec<_>>();
                match args.as_slice() {
                    ["list"] | [] => match self.open_blacklist_editor() {
                        Ok(path) => {
                            self.push_alert(
                                AlertKind::Success,
                                "Blacklist opened",
                                &format!("Opened in editor: {}", path.display()),
                            );
                            self.push_system_message(&format!(
                                "Opened blacklist file in editor:\n{}",
                                path.display()
                            ));
                        }
                        Err(e) => {
                            self.push_alert(
                                AlertKind::Error,
                                "Blacklist open failed",
                                &e.to_string(),
                            );
                            self.push_system_message(&format!("Could not open blacklist file: {e}"));
                        }
                    },
                    ["add", command] => {
                        self.settings.blocked_commands.insert((*command).to_string());
                        let _ = SettingsStore::save(&self.settings);
                        self.messages.push(Message {
                            role: "system".to_string(),
                            content: format!("Blocked command added: {command}"),
                        });
                    }
                    ["remove", command] => {
                        self.settings.blocked_commands.remove(*command);
                        let _ = SettingsStore::save(&self.settings);
                        self.messages.push(Message {
                            role: "system".to_string(),
                            content: format!("Blocked command removed: {command}"),
                        });
                    }
                    _ => self.messages.push(Message {
                        role: "system".to_string(),
                        content: "Usage: /blacklist [list|add <cmd>|remove <cmd>]"
                            .to_string(),
                    }),
                }
            }
            "/new" => {
                self.session_id = generate_session_id();
                self.messages.clear();
                self.messages.push(Message {
                    role: "system".to_string(),
                    content: format!("Started new session: {}", self.session_id),
                });
                self.messages.push(Message {
                    role: "system".to_string(),
                    content: "No past conversation is loaded in this session.".to_string(),
                });
                self.last_persisted_messages_len = 0;
                self.last_status = "New clean session started".to_string();
            }
            "/sessions" => {
                let args = parts.collect::<Vec<_>>();
                if args.is_empty() {
                    self.sessions_menu = list_sessions();
                    self.sessions_selected = 0;
                    self.show_sessions_popup = true;
                    self.last_status = "Sessions menu opened".to_string();
                } else {
                    self.messages.push(Message {
                        role: "system".to_string(),
                        content: "Usage: /sessions".to_string(),
                    });
                }
            }
            "/clear" => {
                self.messages.clear();
                self.messages.push(Message {
                    role: "system".to_string(),
                    content: "Chat cleared.".to_string(),
                });
            }
            "/refresh" => {
                self.refresh_file_tree();
                self.messages.push(Message {
                    role: "system".to_string(),
                    content: "File tree refreshed.".to_string(),
                });
            }
            "/remember" => {
                let text = parts.collect::<Vec<_>>().join(" ");
                if text.is_empty() {
                    self.messages.push(Message {
                        role: "system".to_string(),
                        content: "Usage: /remember <fact>".to_string(),
                    });
                } else {
                    let _ = self.rag.remember_scoped(
                        &format!("session:{}:global", self.session_id),
                        "user",
                        &text,
                    );
                    self.messages.push(Message {
                        role: "system".to_string(),
                        content: "Saved to long-term memory.".to_string(),
                    });
                }
            }
            "/memories" => {
                let q = parts.collect::<Vec<_>>().join(" ");
                let query = if q.is_empty() { "user" } else { &q };
                let items = self
                    .rag
                    .retrieve_scoped(&format!("session:{}:global", self.session_id), query, 5)
                    .unwrap_or_default();
                self.messages.push(Message {
                    role: "system".to_string(),
                    content: if items.is_empty() {
                        "No memories found.".to_string()
                    } else {
                        format!("Memories:\n- {}", items.join("\n- "))
                    },
                });
            }
            "/deep" => {
                let arg = parts.next().unwrap_or_default();
                match arg {
                    "on" => {
                        self.deep_think_enabled = true;
                    }
                    "off" => {
                        self.deep_think_enabled = false;
                    }
                    "" => {}
                    _ => {
                        if let Ok(level) = arg.parse::<u8>() {
                            self.deep_think_level = level.min(10);
                            self.deep_think_enabled = true;
                        }
                    }
                }
                self.messages.push(Message {
                    role: "system".to_string(),
                    content: format!(
                        "Deep Think: {} at {}/10. Disclaimer: higher values consume more tokens.",
                        if self.deep_think_enabled { "ON" } else { "OFF" },
                        self.deep_think_level
                    ),
                });
            }
            "/expand" => {
                let arg = parts.next().unwrap_or_default();
                if arg == "on" {
                    self.coding_expanded = true;
                } else if arg == "off" {
                    self.coding_expanded = false;
                } else {
                    self.coding_expanded = !self.coding_expanded;
                }
                self.messages.push(Message {
                    role: "system".to_string(),
                    content: format!(
                        "Coding mode is now {}. Ctrl+O toggles this too.",
                        if self.coding_expanded {
                            "expanded"
                        } else {
                            "compact"
                        }
                    ),
                });
            }
            "/voice" => {
                self.toggle_voice_recording();
            }
            "/migration" => {
                let arg = parts.next().unwrap_or_default();
                match MigrationSource::from_arg(arg) {
                    Some(source) => self.run_migration_source(source),
                    None => self.push_system_message(
                        "Usage: /migration <claude|openclaw|hermes> (also in Settings tab)",
                    ),
                }
            }
            "/anthropic" => {
                let args = parts.collect::<Vec<_>>();
                match args.as_slice() {
                    ["models", model] => match anthropic_model_id(model) {
                        Some(id) => {
                            self.settings.provider = Provider::Anthropic;
                            self.settings.model = id.to_string();
                            self.settings.enabled_providers = vec![Provider::Anthropic];
                            if let Err(e) = SettingsStore::save(&self.settings) {
                                self.last_status = format!("Settings save failed: {e}");
                            }
                            self.push_system_message(&format!(
                                "Anthropic model set to {id} (provider switched to Anthropic)."
                            ));
                        }
                        None => self.push_system_message(
                            "Unknown model. Options: opus-4.7, sonnet-4.7, haiku-4.7",
                        ),
                    },
                    ["models"] | [] => self.push_system_message(&format!(
                        "Anthropic models: opus-4.7, sonnet-4.7, haiku-4.7. Current: {}. Usage: /anthropic models <model>",
                        self.settings.model
                    )),
                    _ => self.push_system_message(
                        "Usage: /anthropic models <opus-4.7|sonnet-4.7|haiku-4.7>",
                    ),
                }
            }
            "/plugins" => {
                let args = parts.collect::<Vec<_>>();
                match args.as_slice() {
                    [] | ["list"] => {
                        if self.settings.plugins.is_empty() {
                            self.push_system_message(
                                "No plugins tracked. Install with /plugins <name>, browse with /plugins available [query].",
                            );
                        } else {
                            self.push_system_message(&format!(
                                "Installed plugins ({}):\n- {}",
                                self.settings.plugins.len(),
                                self.settings.plugins.join("\n- ")
                            ));
                        }
                    }
                    ["available"] => {
                        let count = plugins::available().len();
                        self.push_system_message(&format!(
                            "{count} plugins available in your Claude Code marketplace. Filter with /plugins available <query>; install with /plugins <name>.",
                        ));
                    }
                    ["available", query] => {
                        let q = query.to_lowercase();
                        let hits: Vec<String> = plugins::available()
                            .into_iter()
                            .filter(|p| p.to_lowercase().contains(q.as_str()))
                            .take(40)
                            .collect();
                        self.push_system_message(&if hits.is_empty() {
                            format!("No catalog plugins match '{query}'.")
                        } else {
                            format!("Matches for '{query}':\n- {}", hits.join("\n- "))
                        });
                    }
                    ["idk"] => self.open_claude_marketplace(),
                    [name] => self.install_plugin(name),
                    _ => self.push_system_message(
                        "Usage: /plugins [list] | available [query] | <name>",
                    ),
                }
            }
            "/plugout" => {
                let args = parts.collect::<Vec<_>>();
                match args.as_slice() {
                    [name] => self.remove_plugin(name),
                    _ => self.push_system_message("Usage: /plugout <name>"),
                }
            }
            "/beetle" => self.start_beetle(),
            "/bots" => {
                let args = parts.collect::<Vec<_>>();
                match args.as_slice() {
                    [] | ["status"] => {
                        let running = self.telegram_stop.is_some();
                        let has_token = self
                            .settings
                            .telegram_bot_token
                            .as_ref()
                            .is_some_and(|t| !t.trim().is_empty());
                        self.push_system_message(&format!(
                            "Telegram bot: {}. Token {}. Paste your token with /bots <token>; also: /bots stop | clear",
                            if running { "running" } else { "stopped" },
                            if has_token { "saved" } else { "not set" },
                        ));
                    }
                    ["start"] | ["on"] => self.start_telegram_bot(),
                    ["stop"] | ["off"] => self.stop_telegram_bot(),
                    ["clear"] | ["remove"] => {
                        self.stop_telegram_bot();
                        self.settings.telegram_bot_token = None;
                        let _ = SettingsStore::save(&self.settings);
                        self.push_system_message("Telegram token cleared.");
                    }
                    [token] => {
                        if self.telegram_stop.is_some() {
                            self.stop_telegram_bot();
                        }
                        self.settings.telegram_bot_token = Some((*token).to_string());
                        if let Err(e) = SettingsStore::save(&self.settings) {
                            self.push_alert(AlertKind::Error, "Settings save failed", &e.to_string());
                        }
                        self.push_system_message("Telegram token saved (encrypted). Starting bot…");
                        self.start_telegram_bot();
                    }
                    _ => self.push_system_message(
                        "Usage: /bots <token> | start | stop | clear | status",
                    ),
                }
            }
            _ => {
                self.push_alert(
                    AlertKind::Error,
                    "Unknown slash command",
                    &format!("{name} is not supported"),
                );
                self.push_system_message("Unknown command");
            }
        }
    }

    fn toggle_voice_recording(&mut self) {
        if self.voice_transcribing {
            self.last_status = "Finishing transcription…".to_string();
            return;
        }
        if self.voice_recorder.is_some() {
            self.stop_and_transcribe_voice();
        } else {
            let cfg = match CloudSttConfig::from_settings(&self.settings) {
                Ok(c) => c,
                Err(e) => {
                    self.last_status = format!("{e:#}");
                    self.push_alert(
                        AlertKind::Error,
                        "Push-to-talk unavailable",
                        &format!("{e:#}"),
                    );
                    return;
                }
            };
            match VoiceRecorder::start() {
                Ok(rec) => {
                    let sample_rate = rec.sample_rate();
                    let buf = rec.sample_buffer();
                    self.voice_level = rec.level.clone();
                    let active = Arc::new(AtomicBool::new(true));
                    let active_bg = active.clone();
                    let (tx, rx) = mpsc::channel();
                    let cfg_bg = cfg.clone();
                    std::thread::spawn(move || {
                        spawn_live_stt(buf, sample_rate, active_bg, cfg_bg, tx);
                    });
                    self.voice_live_active = Some(active);
                    self.voice_rx = Some(rx);
                    self.voice_partial.clear();
                    self.voice_recorder = Some(rec);
                    self.last_status = format!(
                        "🎤 Push-to-talk active ({}) — hold F9 1s to start · Alt+V to stop",
                        cfg.backend_label()
                    );
                    self.push_alert(
                        AlertKind::Info,
                        "Push-to-talk started",
                        "Speaking is active (hold F9 1s to start). Press Alt+V to stop and insert transcript into input.",
                    );
                }
                Err(e) => {
                    self.last_status = format!("Voice start failed: {e}");
                    self.push_alert(
                        AlertKind::Error,
                        "Microphone start failed",
                        &e.to_string(),
                    );
                }
            }
        }
    }

    fn stop_and_transcribe_voice(&mut self) {
        let Some(rec) = self.voice_recorder.take() else {
            return;
        };
        if let Err(e) = rec.finish() {
            self.last_status = format!("Voice capture failed: {e}");
            if let Some(a) = self.voice_live_active.take() {
                a.store(false, Ordering::SeqCst);
            }
            return;
        }
        if let Some(a) = self.voice_live_active.take() {
            a.store(false, Ordering::SeqCst);
        }
        self.voice_transcribing = true;
        self.last_status = "🎙️ Finalizing transcript…".to_string();
    }

    fn sync_voice_state(&mut self) {
        self.try_start_voice_after_f9_hold();
        if self.voice_recorder.is_some() {
            let lvl = self.voice_level.load(Ordering::Relaxed);
            let hint = if self.voice_partial.is_empty() {
                String::new()
            } else {
                " · streaming".to_string()
            };
            self.last_status = format!("{}{hint}", render_level_bar(lvl));
        } else if self.voice_transcribing {
            self.last_status = "🎙️ Finalizing transcript…".to_string();
        }
        if let Some(rx) = self.voice_rx.as_ref() {
            loop {
                match rx.try_recv() {
                    Ok(VoiceSttUpdate::Partial(t)) => {
                        self.voice_partial = t;
                    }
                    Ok(VoiceSttUpdate::Final(res)) => {
                        self.voice_transcribing = false;
                        self.voice_rx = None;
                        self.voice_partial.clear();
                        match res {
                            Ok(text) => {
                                if text.is_empty() {
                                    self.last_status = "Transcription was empty.".to_string();
                                } else {
                                    if !self.input.is_empty() && !self.input.ends_with(' ') {
                                        self.input.push(' ');
                                    }
                                    self.input.push_str(text.trim());
                                    self.last_status = "✍️ Voice added to input.".to_string();
                                }
                            }
                            Err(e) => {
                                self.last_status = format!("Transcription failed: {e:#}");
                                self.push_alert(
                                    AlertKind::Error,
                                    "Transcription failed",
                                    &format!("{e:#}"),
                                );
                            }
                        }
                        break;
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        self.voice_transcribing = false;
                        self.voice_rx = None;
                        self.voice_partial.clear();
                        self.last_status =
                            "Voice STT thread ended unexpectedly.".to_string();
                        self.push_alert(
                            AlertKind::Error,
                            "Voice STT thread ended",
                            "The streaming transcription worker disconnected unexpectedly.",
                        );
                        break;
                    }
                }
            }
        }
    }

    fn persist_session_if_needed(&mut self) {
        if self.messages.len() == self.last_persisted_messages_len {
            return;
        }
        if save_session_messages(&self.session_id, &self.messages).is_ok() {
            self.last_persisted_messages_len = self.messages.len();
        }
    }

    /// Set the terminal tab/window title to "⬡ <current task>", à la Claude Code. Only emits
    /// the OSC escape when the title actually changes, to avoid per-tick churn.
    fn sync_terminal_title(&mut self) {
        let title = self.compute_tab_title();
        if title != self.tab_title {
            let _ = execute!(std::io::stdout(), SetTitle(&title));
            self.tab_title = title;
        }
    }

    fn compute_tab_title(&self) -> String {
        const LOGO: &str = "⬡"; // OpenCage mark (a "cell/cage"), distinct from Claude Code's ✳.
        if self.busy {
            return format!("{LOGO} OpenCage · working…");
        }
        let mut summary = summarize_session_title(&self.messages).replace('\n', " ");
        if summary.chars().count() > 40 {
            summary = summary.chars().take(39).collect::<String>() + "…";
        }
        if summary.is_empty() || summary == "New session" {
            format!("{LOGO} OpenCage")
        } else {
            format!("{LOGO} {summary}")
        }
    }

    /// Spawn the Telegram polling thread (no-op if already running or no token saved).
    fn start_telegram_bot(&mut self) {
        if self.telegram_stop.is_some() {
            self.push_system_message("Telegram bot is already running.");
            return;
        }
        let Some(token) = self
            .settings
            .telegram_bot_token
            .clone()
            .filter(|t| !t.trim().is_empty())
        else {
            self.push_system_message("No Telegram token set. Use /bots <token>.");
            return;
        };
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel();
        let settings = self.settings.clone();
        let cwd = self.current_dir.clone();
        let stop_bg = stop.clone();
        std::thread::spawn(move || telegram::run_bot(token, settings, cwd, stop_bg, tx));
        self.telegram_stop = Some(stop);
        self.telegram_rx = Some(rx);
        self.last_status = "Telegram bot starting…".to_string();
    }

    fn stop_telegram_bot(&mut self) {
        if let Some(stop) = self.telegram_stop.take() {
            stop.store(true, Ordering::SeqCst);
            self.push_system_message("Telegram bot stopping…");
        } else {
            self.push_system_message("Telegram bot is not running.");
        }
    }

    /// Surface Telegram activity lines in the chat (incoming messages, replies, status).
    fn drain_telegram_log(&mut self) {
        let mut lines = Vec::new();
        if let Some(rx) = self.telegram_rx.as_ref() {
            while let Ok(line) = rx.try_recv() {
                lines.push(line);
            }
        }
        for line in lines {
            self.push_system_message(&line);
        }
    }

    fn load_session_messages(&mut self) -> Result<()> {
        self.messages = load_session_messages(&self.session_id)?;
        self.last_persisted_messages_len = self.messages.len();
        Ok(())
    }

    fn switch_to_selected_session(&mut self) {
        let Some(meta) = self.sessions_menu.get(self.sessions_selected).cloned() else {
            self.show_sessions_popup = false;
            return;
        };
        self.session_id = meta.id.clone();
        self.messages.clear();
        if let Err(e) = self.load_session_messages() {
            self.messages.push(Message {
                role: "system".to_string(),
                content: format!("Failed to switch session: {e}"),
            });
        }
        if self.messages.is_empty() {
            self.messages.push(Message {
                role: "system".to_string(),
                content: "Switched session is empty.".to_string(),
            });
        }
        self.show_sessions_popup = false;
        self.last_status = format!("Switched to session {}", meta.id);
    }

    /// Permanently delete the session at `idx` in the open sessions popup.
    fn delete_session_at(&mut self, idx: usize) {
        let Some(meta) = self.sessions_menu.get(idx).cloned() else {
            return;
        };
        match delete_session_file(&meta.id) {
            Ok(()) => {
                self.sessions_menu.remove(idx);
                if self.sessions_selected >= self.sessions_menu.len() {
                    self.sessions_selected = self.sessions_menu.len().saturating_sub(1);
                }
                self.push_alert(
                    AlertKind::Success,
                    "Session deleted",
                    &format!("Permanently deleted: {}", meta.title),
                );
                self.last_status = format!("Deleted session {}", meta.id);
            }
            Err(e) => {
                self.push_alert(AlertKind::Error, "Delete failed", &e.to_string());
                self.last_status = format!("Delete failed: {e}");
            }
        }
    }

    /// Open the `/model switch` popup listing the providers you've configured.
    fn open_provider_switch(&mut self) {
        let configured: Vec<Provider> = Provider::all()
            .into_iter()
            .filter(|p| provider_has_credentials(&self.settings, p))
            .collect();
        // Fall back to all providers if none have credentials yet, so you can still pick.
        let menu = if configured.is_empty() {
            Provider::all().to_vec()
        } else {
            configured
        };
        self.provider_selected = menu
            .iter()
            .position(|p| std::mem::discriminant(p) == std::mem::discriminant(&self.settings.provider))
            .unwrap_or(0);
        self.provider_menu = menu;
        self.show_provider_popup = true;
        self.last_status = "Pick a provider (↑/↓ + Enter, Esc to close)".to_string();
    }

    fn apply_provider_switch(&mut self) {
        let Some(provider) = self.provider_menu.get(self.provider_selected).cloned() else {
            self.show_provider_popup = false;
            return;
        };
        // Keep a remembered model for this provider if valid; else use its default.
        let model = self
            .settings
            .provider_models
            .get(provider.as_str())
            .cloned()
            .filter(|m| provider.models().iter().any(|x| *x == m.as_str()))
            .unwrap_or_else(|| provider.models().first().copied().unwrap_or("").to_string());
        self.settings.provider = provider.clone();
        self.settings.model = model;
        // Forget any previously-selected provider — only the chosen one stays active.
        self.settings.enabled_providers = vec![provider.clone()];
        if let Err(e) = SettingsStore::save(&self.settings) {
            self.last_status = format!("Settings save failed: {e}");
        }
        self.show_provider_popup = false;
        self.push_system_message(&format!(
            "Switched to {} ({}).",
            provider.as_str(),
            self.settings.model
        ));
        self.last_status = format!("Provider: {}", provider.as_str());
    }

    fn open_blacklist_editor(&self) -> Result<PathBuf> {
        let mut commands: Vec<_> = self.settings.blocked_commands.iter().cloned().collect();
        commands.sort();
        let mut lines = vec![
            "# Opencage Blacklisted Commands".to_string(),
            "# One command per line".to_string(),
            "# Lines starting with # are comments".to_string(),
            String::new(),
        ];
        lines.push("# Active blacklist".to_string());
        lines.extend(commands);
        lines.push(String::new());

        let config_dir = dirs::config_dir()
            .context("Failed to resolve config directory")?
            .join("opencage");
        fs::create_dir_all(&config_dir)?;
        let path = config_dir.join("blacklist.txt");
        fs::write(&path, lines.join("\n"))?;

        let path_str = path.to_string_lossy().to_string();
        if let Ok(editor) = std::env::var("OPENCAGE_BLACKLIST_EDITOR") {
            let editor = editor.trim();
            if !editor.is_empty() && Command::new(editor).arg(&path_str).spawn().is_ok() {
                return Ok(path);
            }
        }
        if let Ok(editor) = std::env::var("EDITOR") {
            let editor = editor.trim();
            if !editor.is_empty() && Command::new(editor).arg(&path_str).spawn().is_ok() {
                return Ok(path);
            }
        }
        if let Some(found) = find_notepad_under_user_app_folders() {
            if Command::new(&found).arg(&path_str).spawn().is_ok() {
                return Ok(path);
            }
        }
        // Prefer Notepad / GUI editors; avoid opening nano first.
        for editor in [
            "notepad",
            "gedit",
            "mousepad",
            "xed",
            "kate",
            "leafpad",
            "geany",
            "featherpad",
            "kwrite",
            "code",
            "cursor",
        ] {
            if Command::new(editor).arg(&path_str).spawn().is_ok() {
                return Ok(path);
            }
        }
        if Command::new("xdg-open").arg(&path_str).spawn().is_ok() {
            return Ok(path);
        }
        for editor in ["micro", "nvim", "vim", "vi", "nano"] {
            if Command::new(editor).arg(&path_str).spawn().is_ok() {
                return Ok(path);
            }
        }
        Err(anyhow::anyhow!(
            "No editor found. Put a Notepad-like app under ~/Apps, ~/apps, or ~/Applications (name contains \"notepad\"), or set OPENCAGE_BLACKLIST_EDITOR / EDITOR."
        ))
    }

    fn blacklist_file_path() -> Result<PathBuf> {
        Ok(dirs::config_dir()
            .context("Failed to resolve config directory")?
            .join("opencage")
            .join("blacklist.txt"))
    }

    fn sync_blacklist_file(&mut self) {
        let Ok(path) = Self::blacklist_file_path() else {
            return;
        };
        let Ok(meta) = fs::metadata(&path) else {
            return;
        };
        let Ok(modified) = meta.modified() else {
            return;
        };
        if self
            .blacklist_last_modified
            .is_some_and(|seen| seen >= modified)
        {
            return;
        }

        let Ok(content) = fs::read_to_string(&path) else {
            return;
        };
        let mut next = HashSet::new();
        for line in content.lines() {
            let cmd = line.trim();
            if cmd.is_empty() || cmd.starts_with('#') {
                continue;
            }
            next.insert(cmd.to_string());
        }

        if next != self.settings.blocked_commands {
            self.settings.blocked_commands = next;
            if let Err(e) = SettingsStore::save(&self.settings) {
                self.last_status = format!("Blacklist sync save failed: {e}");
            } else {
                self.last_status = "Reloaded blacklist.txt".to_string();
            }
        }
        self.blacklist_last_modified = Some(modified);
    }

    fn refresh_file_tree(&mut self) {
        let root = self.current_dir.clone();
        let root_expanded = self.file_tree_root_expanded;
        let mut nodes = vec![FileNode {
            path: root.clone(),
            depth: 0,
            is_dir: true,
            expanded: root_expanded,
        }];
        if root_expanded {
            self.collect_nodes(&root, 1, &mut nodes);
        }
        self.files = nodes;
        if self.selected_file_idx >= self.files.len() {
            self.selected_file_idx = self.files.len().saturating_sub(1);
        }
    }

    fn sync_cwd_and_tree(&mut self) {
        let now = Instant::now();
        let cwd = std::env::current_dir().unwrap_or_else(|_| self.current_dir.clone());
        if cwd != self.current_dir {
            self.current_dir = cwd.clone();
            let key = cwd.to_string_lossy().to_string();
            self.current_dir_trusted = self.settings.trusted_paths.contains(&key);
            self.expanded_paths.clear();
            self.file_tree_root_expanded = true;
            self.refresh_file_tree();
            self.last_status = format!("Switched root to {}", cwd.display());
            self.last_tree_refresh = now;
            return;
        }
        if now.duration_since(self.last_tree_refresh) >= Duration::from_millis(700) {
            self.refresh_file_tree();
            self.last_tree_refresh = now;
        }
    }

    fn start_autocode(&mut self, prompt: String) {
        self.busy = true;
        self.last_status = "Autonomous coding in progress...".to_string();
        let settings = self.settings.clone();
        let history = self.messages.clone();
        let deep_think_enabled = self.deep_think_enabled;
        let deep_think_level = self.deep_think_level;
        let coding_expanded = self.coding_expanded;
        let cwd = self.current_dir.clone();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let result = run_autonomous_coding(
                &settings,
                &history,
                &prompt,
                deep_think_enabled,
                deep_think_level,
                coding_expanded,
                &cwd,
            )
            .unwrap_or_else(|e| AgentResult {
                text: format!("Autocode failed: {e}"),
                commands: vec![],
            });
            let _ = tx.send(result);
        });
        self.pending_response = Some(rx);
    }

    fn prompt_next_command_if_needed(&mut self) {
        if self.pending_command.is_some() {
            return;
        }
        if let Some(cmd) = self.queued_commands.pop_front() {
            self.pending_command = Some(cmd);
            self.command_approval_yes = true;
            self.last_status = "Command approval required".to_string();
            if let Some(pending) = self.pending_command.as_ref() {
                self.push_alert(
                    AlertKind::Warning,
                    "Command approval requested",
                    &format!("Approve or deny: {pending}"),
                );
            }
        }
    }

    fn apply_pending_command_choice(&mut self, approved: bool) {
        let Some(cmd) = self.pending_command.take() else {
            return;
        };
        if approved {
            match run_in_sandbox(&cmd, &self.settings, true) {
                Ok(_) => self.push_alert(
                    AlertKind::Success,
                    "Command executed",
                    &cmd,
                ),
                Err(e) => self.push_alert(
                    AlertKind::Error,
                    "Command execution failed",
                    &format!("{cmd} -> {e}"),
                ),
            }
            self.last_status = "command approved".to_string();
        } else {
            self.push_alert(AlertKind::Info, "Command denied", &cmd);
            self.last_status = "command denied".to_string();
        }
        self.prompt_next_command_if_needed();
    }

    fn collect_nodes(&self, root: &Path, depth: usize, target: &mut Vec<FileNode>) {
        let mut entries: Vec<_> = WalkDir::new(root)
            .min_depth(1)
            .max_depth(1)
            .into_iter()
            .filter_map(|e| e.ok())
            .collect();
        entries.sort_by_key(|e| e.path().to_path_buf());
        for e in entries {
            let p = e.path().to_path_buf();
            if p.file_name().and_then(|n| n.to_str()) == Some("target") {
                continue;
            }
            let is_dir = p.is_dir();
            let expanded = is_dir && self.expanded_paths.contains(&p);
            target.push(FileNode {
                path: p.clone(),
                depth,
                is_dir,
                expanded,
            });
            if expanded {
                self.collect_nodes(&p, depth + 1, target);
            }
        }
    }

    fn toggle_node(&mut self) {
        if self.files.is_empty() {
            return;
        }
        let selected = self.files[self.selected_file_idx].clone();
        if !selected.is_dir {
            return;
        }
        // Synthetic root row: controls whether top-level entries are listed at all.
        if selected.path == self.current_dir {
            self.file_tree_root_expanded = !self.file_tree_root_expanded;
            if !self.file_tree_root_expanded {
                self.expanded_paths.clear();
            }
            self.refresh_file_tree();
            return;
        }
        if selected.expanded {
            let base_depth = selected.depth;
            let i = self.selected_file_idx + 1;
            while i < self.files.len() && self.files[i].depth > base_depth {
                self.files.remove(i);
            }
            self.files[self.selected_file_idx].expanded = false;
            self.expanded_paths
                .retain(|p| !p.starts_with(&selected.path));
        } else {
            let mut insert_at = self.selected_file_idx + 1;
            let mut entries: Vec<_> = WalkDir::new(&selected.path)
                .min_depth(1)
                .max_depth(1)
                .into_iter()
                .filter_map(|e| e.ok())
                .collect();
            entries.sort_by_key(|e| e.path().to_path_buf());
            for e in entries {
                let p = e.path().to_path_buf();
                if p.file_name().and_then(|n| n.to_str()) == Some("target") {
                    continue;
                }
                self.files.insert(
                    insert_at,
                    FileNode {
                        path: p.clone(),
                        depth: selected.depth + 1,
                        is_dir: p.is_dir(),
                        expanded: false,
                    },
                );
                insert_at += 1;
            }
            self.files[self.selected_file_idx].expanded = true;
            self.expanded_paths.insert(selected.path);
        }
    }

    fn open_settings_tab(&mut self) {
        let providers = Provider::all();
        let provider_idx = providers
            .iter()
            .position(|p| std::mem::discriminant(p) == std::mem::discriminant(&self.settings.provider))
            .unwrap_or(1);
        let provider = &providers[provider_idx];
        let model_idx = provider
            .models()
            .iter()
            .position(|m| *m == self.settings.model)
            .unwrap_or(0);
        self.settings_form = Some(SettingsForm {
            selected: 0,
            provider_idx,
            model_idx,
            avatar: self.settings.ai_avatar.clone(),
            cuss_filter: self.settings.cuss_filter,
            openai: self.settings.openai_api_key.clone().unwrap_or_default(),
            groq: self.settings.groq_api_key.clone().unwrap_or_default(),
            anthropic: self.settings.anthropic_api_key.clone().unwrap_or_default(),
            moonshot: self.settings.moonshot_api_key.clone().unwrap_or_default(),
            glm: self.settings.glm_api_key.clone().unwrap_or_default(),
            copilot: self.settings.github_copilot_token.clone().unwrap_or_default(),
            migration_idx: 0,
            report: vec!["Press Ctrl+S to save, F5 to validate, Esc to close".to_string()],
        });
        self.active_tab = ActiveTab::Settings;
        self.last_status = "Settings tab opened".to_string();
    }

    fn settings_form_to_settings(&self, form: &SettingsForm) -> Settings {
        let mut s = self.settings.clone();
        s.provider = Provider::all()[form.provider_idx].clone();
        s.model = s.provider.models()[form.model_idx].to_string();
        // Only the selected provider stays active — don't keep prior providers connected.
        s.enabled_providers = vec![s.provider.clone()];
        s.ai_avatar = form.avatar.clone();
        s.cuss_filter = form.cuss_filter;
        s.openai_api_key = if form.openai.trim().is_empty() {
            None
        } else {
            Some(form.openai.clone())
        };
        s.groq_api_key = if form.groq.trim().is_empty() {
            None
        } else {
            Some(form.groq.clone())
        };
        s.anthropic_api_key = if form.anthropic.trim().is_empty() {
            None
        } else {
            Some(form.anthropic.clone())
        };
        s.moonshot_api_key = if form.moonshot.trim().is_empty() {
            None
        } else {
            Some(form.moonshot.clone())
        };
        s.glm_api_key = if form.glm.trim().is_empty() {
            None
        } else {
            Some(form.glm.clone())
        };
        s.github_copilot_token = if form.copilot.trim().is_empty() {
            None
        } else {
            Some(form.copilot.clone())
        };
        s
    }

    fn validate_settings_form(&mut self) {
        if let Some(mut form) = self.settings_form.take() {
            let candidate = self.settings_form_to_settings(&form);
            form.report = validate_settings_keys(&candidate);
            self.last_status = "Validation complete".to_string();
            self.settings_form = Some(form);
        }
    }

    fn save_settings_form(&mut self) {
        if let Some(mut form) = self.settings_form.take() {
            let candidate = self.settings_form_to_settings(&form);
            form.report = validate_settings_keys(&candidate);
            match SettingsStore::save(&candidate) {
                Ok(()) => {
                    self.settings = candidate;
                    self.last_status = format!("Saved settings to {}", SettingsStore::path().display());
                    form.report.push("Saved successfully".to_string());
                }
                Err(e) => {
                    self.last_status = format!("Save failed: {e}");
                    form.report.push(format!("Save failed: {e}"));
                }
            }
            self.settings_form = Some(form);
        }
    }

    /// Import history (and, for Claude Code, the OAuth token) from another AI CLI.
    /// The OAuth token is stored in its own settings field; existing API keys are left alone.
    fn run_migration_source(&mut self, source: MigrationSource) {
        self.last_status = format!("Migrating from {}…", source.label());
        let summary = match migration::migrate(source) {
            Ok(outcome) => {
                let mut imported = 0usize;
                for s in &outcome.sessions {
                    let id = format!("{}-{}", source.id_prefix(), s.source_id);
                    if save_imported_session(&id, &s.messages, s.updated_ts).is_ok() {
                        imported += 1;
                    }
                }
                let oauth_captured = outcome.oauth.is_some();
                if let Some(creds) = outcome.oauth {
                    self.settings.anthropic_oauth_token = Some(creds.access_token);
                    self.settings.anthropic_oauth_expires_at = creds.expires_at;
                }
                // Claude Code also exposes its installed plugins — mirror them into OpenCage.
                let mut synced_plugins = 0usize;
                if source == MigrationSource::ClaudeCode {
                    self.settings.plugins = plugins::read_installed();
                    synced_plugins = self.settings.plugins.len();
                }
                if let Err(e) = SettingsStore::save(&self.settings) {
                    self.push_alert(AlertKind::Error, "Settings save failed", &e.to_string());
                }
                let msg = format!(
                    "Migration from {}: imported {imported} session(s){}{}.",
                    source.label(),
                    if oauth_captured {
                        ", captured OAuth token (API-key settings unchanged)"
                    } else {
                        ""
                    },
                    if synced_plugins > 0 {
                        format!(", synced {synced_plugins} plugin(s)")
                    } else {
                        String::new()
                    }
                );
                self.push_alert(AlertKind::Success, "Migration complete", &msg);
                msg
            }
            Err(e) => {
                let msg = format!("Migration from {} failed: {e}", source.label());
                self.push_alert(AlertKind::Error, "Migration failed", &msg);
                msg
            }
        };
        self.push_system_message(&summary);
        self.last_status = summary.clone();
        if let Some(form) = self.settings_form.as_mut() {
            form.report = vec![summary];
        }
    }

    /// Install + enable a marketplace plugin, syncing it into Claude Code and tracking it here.
    fn install_plugin(&mut self, name: &str) {
        match plugins::install(name) {
            Ok(id) => {
                if !self.settings.plugins.contains(&id) {
                    self.settings.plugins.push(id.clone());
                    self.settings.plugins.sort();
                }
                if let Err(e) = SettingsStore::save(&self.settings) {
                    self.push_alert(AlertKind::Error, "Settings save failed", &e.to_string());
                }
                self.push_alert(
                    AlertKind::Success,
                    "Plugin installed",
                    &format!("{id} (synced to Claude Code)"),
                );
                self.push_system_message(&format!(
                    "Installed plugin {id} and enabled it in Claude Code."
                ));
            }
            Err(e) => {
                self.push_alert(AlertKind::Error, "Plugin install failed", &e.to_string());
                self.push_system_message(&format!("Could not install plugin: {e}"));
            }
        }
    }

    /// Remove + disable a plugin, in both OpenCage and Claude Code.
    fn remove_plugin(&mut self, name: &str) {
        match plugins::uninstall(name) {
            Ok(id) => {
                self.settings.plugins.retain(|p| p != &id);
                if let Err(e) = SettingsStore::save(&self.settings) {
                    self.push_alert(AlertKind::Error, "Settings save failed", &e.to_string());
                }
                self.push_alert(
                    AlertKind::Success,
                    "Plugin removed",
                    &format!("{id} (synced to Claude Code)"),
                );
                self.push_system_message(&format!(
                    "Removed plugin {id} from OpenCage and Claude Code."
                ));
            }
            Err(e) => {
                self.push_alert(AlertKind::Error, "Plugin remove failed", &e.to_string());
                self.push_system_message(&format!("Could not remove plugin: {e}"));
            }
        }
    }

    /// `/plugins idk`: open a fresh terminal window running Claude Code's interactive
    /// plugin marketplace (`claude /plugin`). The window can simply be closed when done.
    fn open_claude_marketplace(&mut self) {
        let inner = "claude /plugin";
        // Tried in order; first emulator that spawns wins. `bash -lc` gives a login PATH
        // so `claude` (in ~/.local/bin) resolves regardless of how OpenCage was launched.
        let attempts: [(&str, &[&str]); 6] = [
            ("gnome-terminal", &["--", "bash", "-lc", inner]),
            ("konsole", &["-e", "bash", "-lc", inner]),
            ("x-terminal-emulator", &["-e", "bash", "-lc", inner]),
            ("alacritty", &["-e", "bash", "-lc", inner]),
            ("kitty", &["bash", "-lc", inner]),
            ("xterm", &["-e", "bash", "-lc", inner]),
        ];
        for (term, args) in attempts {
            if Command::new(term).args(args).spawn().is_ok() {
                self.push_alert(
                    AlertKind::Success,
                    "Opening marketplace",
                    &format!("Launched {term} running `{inner}`."),
                );
                self.push_system_message(&format!(
                    "Opened a new {term} window running `{inner}` (Claude Code plugin marketplace). Close it when done."
                ));
                self.last_status = "Opened Claude plugin marketplace".to_string();
                return;
            }
        }
        self.push_alert(
            AlertKind::Error,
            "No terminal found",
            "Could not launch a terminal (tried gnome-terminal, konsole, x-terminal-emulator, alacritty, kitty, xterm).",
        );
        self.push_system_message("Could not open a terminal window for the marketplace.");
    }

    /// `/beetle`: start the Beetle Design web server and open it in the browser.
    fn start_beetle(&mut self) {
        if let Some(stop) = self.beetle_stop.clone() {
            if !stop.load(Ordering::Relaxed) {
                if let Some(url) = self.beetle_url.clone() {
                    open_in_browser(&url);
                    self.push_system_message(&format!("Beetle Design is already running: {url}"));
                }
                return;
            }
            // The previous server stopped itself (tab closed) — clear and start fresh.
            self.beetle_stop = None;
            self.beetle_url = None;
        }
        let listener = match std::net::TcpListener::bind(("127.0.0.1", 0)) {
            Ok(l) => l,
            Err(e) => {
                self.push_alert(AlertKind::Error, "Beetle failed to start", &e.to_string());
                return;
            }
        };
        let port = listener.local_addr().map(|a| a.port()).unwrap_or(0);
        let url = format!("http://127.0.0.1:{port}/");
        let cwd = self.current_dir.clone();
        let rag = self.rag.clone();
        let session_id = self.session_id.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_bg = stop.clone();
        std::thread::spawn(move || {
            crate::ai::beetle::serve(listener, cwd, rag, session_id, stop_bg)
        });
        self.beetle_stop = Some(stop);
        self.beetle_url = Some(url.clone());
        open_in_browser(&url);
        self.push_alert(AlertKind::Success, "Beetle Design", &format!("Live design UI at {url}"));
        self.push_system_message(&format!(
            "Beetle Design is live at {url} (opened in your browser). Describe your app, preview it, then push or download. Closing the tab stops the server."
        ));
        self.last_status = "Beetle Design running".to_string();
    }

    /// Notice when the Beetle server has stopped itself (tab closed / heartbeat lost) so `/beetle`
    /// can start a fresh one.
    fn sync_beetle_state(&mut self) {
        let stopped = self
            .beetle_stop
            .as_ref()
            .is_some_and(|s| s.load(Ordering::Relaxed));
        if stopped {
            self.beetle_stop = None;
            self.beetle_url = None;
            self.push_system_message("Beetle Design server stopped (tab closed).");
            self.last_status = "Beetle Design stopped".to_string();
        }
    }

    fn filter_cuss(&self, text: String) -> String {
        if !self.settings.cuss_filter {
            return text;
        }
        let mut sanitized = text;
        for bad in ["fuck", "shit", "bitch", "asshole", "damn"] {
            sanitized = sanitized.replace(bad, "***");
            sanitized = sanitized.replace(&bad.to_uppercase(), "***");
        }
        sanitized
    }

    pub fn filtered_suggestions(&self) -> Vec<&str> {
        let needle = self.input.trim_end().to_lowercase();
        self.palette_commands
            .iter()
            .copied()
            .filter(|cmd| cmd.to_lowercase().starts_with(&needle))
            .collect()
    }

    pub fn context_usage_chars(&self) -> usize {
        self.messages
            .iter()
            .map(|m| m.role.len() + m.content.len() + 8)
            .sum()
    }

    pub fn pending_popup_text(&self) -> Option<String> {
        if self.pending_trust_prompt {
            return Some(format!(
                "Trust folder?\n{}\nType yes/y or no/n",
                self.current_dir.display()
            ));
        }
        None
    }

    pub fn command_approval_bar(&self) -> Option<(String, bool)> {
        self.pending_command
            .as_ref()
            .map(|cmd| (cmd.clone(), self.command_approval_yes))
    }

    pub fn sessions_popup_state(&self) -> Option<(Vec<SessionMeta>, usize)> {
        if self.show_sessions_popup {
            Some((self.sessions_menu.clone(), self.sessions_selected))
        } else {
            None
        }
    }

    /// Display rows + selected index for the `/model switch` popup (None when closed).
    pub fn provider_popup_state(&self) -> Option<(Vec<String>, usize)> {
        if !self.show_provider_popup {
            return None;
        }
        let rows = self
            .provider_menu
            .iter()
            .map(|p| {
                let current = std::mem::discriminant(p)
                    == std::mem::discriminant(&self.settings.provider);
                format!("{}{}", p.as_str(), if current { " (current)" } else { "" })
            })
            .collect();
        Some((rows, self.provider_selected))
    }

    pub fn alerts(&self) -> Vec<AlertItem> {
        self.alerts.iter().cloned().collect()
    }

    pub fn live_tasks(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.busy {
            out.push("Agent task running".to_string());
        }
        if let Some(cmd) = self.pending_command.as_ref() {
            out.push(format!("Waiting command approval: {cmd}"));
        }
        if self.pending_trust_prompt {
            out.push("Waiting folder trust yes/no".to_string());
        }
        if self.voice_recorder.is_some() {
            out.push("Voice capture active".to_string());
        }
        if self.voice_transcribing {
            out.push("Voice transcription in progress".to_string());
        }
        if out.is_empty() {
            out.push("No active tasks".to_string());
        }
        out
    }

    pub fn pending_command_for_alerts(&self) -> Option<(String, bool)> {
        self.pending_command
            .as_ref()
            .map(|cmd| (cmd.clone(), self.command_approval_yes))
    }

    pub fn voice_meter_level(&self) -> u8 {
        self.voice_level.load(Ordering::Relaxed)
    }

    pub fn is_push_to_talk_active(&self) -> bool {
        self.voice_recorder.is_some()
    }

    fn suggestion_row_range(&self) -> Option<(usize, usize)> {
        if !self.show_palette {
            return None;
        }
        let lines = std::env::var("LINES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(40);
        let panel_h = 6usize;
        let input_h = ui::CHAT_FOOTER_HEIGHT as usize;
        let start = lines.saturating_sub(input_h + panel_h);
        let end = start + panel_h.saturating_sub(2);
        Some((start, end))
    }

    fn push_system_message(&mut self, content: &str) {
        self.messages.push(Message {
            role: "system".to_string(),
            content: content.to_string(),
        });
    }

    fn push_alert(&mut self, kind: AlertKind, title: &str, message: &str) {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let at = format!("{ts}");
        self.alerts.push_front(AlertItem {
            at,
            kind,
            title: title.to_string(),
            message: message.to_string(),
        });
        while self.alerts.len() > 120 {
            self.alerts.pop_back();
        }
    }

    /// In the sessions popup: clicking the red circle deletes a session; clicking
    /// elsewhere on a row selects it. Layout mirrors `ui::centered_rect(70, 55, …)`.
    fn try_sessions_delete_click(&mut self, mouse: MouseEvent) {
        let Ok((w, h)) = size() else {
            return;
        };
        let term = Rect::new(0, 0, w, h);
        let area = ui::centered_rect(70, 55, term);
        // Content sits inside the bordered block, offset by one row/column.
        let inner_x = area.x + 1;
        let inner_y = area.y + 1;
        if mouse.row < inner_y {
            return;
        }
        let idx = (mouse.row - inner_y) as usize;
        if idx >= self.sessions_menu.len() {
            return;
        }
        // The red circle "● " occupies the first two content columns.
        if mouse.column >= inner_x && mouse.column <= inner_x + 1 {
            self.delete_session_at(idx);
        } else {
            self.sessions_selected = idx;
        }
    }

    /// Clicking a row in the `/model switch` popup selects and applies that provider.
    fn try_provider_click(&mut self, mouse: MouseEvent) {
        let Ok((w, h)) = size() else {
            return;
        };
        let area = ui::centered_rect(50, 45, Rect::new(0, 0, w, h));
        let inner_y = area.y + 1;
        if mouse.row < inner_y {
            return;
        }
        let idx = (mouse.row - inner_y) as usize;
        if idx < self.provider_menu.len() {
            self.provider_selected = idx;
            self.apply_provider_switch();
        }
    }

    fn try_alerts_command_click(&mut self, mouse: MouseEvent) -> bool {
        let Some((_, _yes_selected)) = self.pending_command_for_alerts() else {
            return false;
        };
        let Ok((w, h)) = size() else {
            return false;
        };
        if h < 6 || w < 30 {
            return false;
        }
        // Footer area mirrors alerts tab layout. Buttons are at fixed horizontal slices.
        let footer_start = h.saturating_sub(ui::CHAT_FOOTER_HEIGHT);
        if mouse.row < footer_start {
            return false;
        }
        let yes_start = w.saturating_mul(65) / 100 + 2;
        let yes_end = yes_start + 6;
        let no_start = yes_end + 2;
        let no_end = no_start + 5;
        if mouse.column >= yes_start && mouse.column <= yes_end {
            self.apply_pending_command_choice(true);
            return true;
        }
        if mouse.column >= no_start && mouse.column <= no_end {
            self.apply_pending_command_choice(false);
            return true;
        }
        false
    }
}

fn user_notepad_search_roots() -> Vec<PathBuf> {
    let Some(home) = dirs::home_dir() else {
        return vec![];
    };
    // `mut` only needed on Linux/Windows extra roots; macOS cross-build warns otherwise.
    #[allow(unused_mut)]
    let mut roots = vec![
        home.join("Apps"),
        home.join("apps"),
        home.join("Applications"),
    ];
    #[cfg(target_os = "linux")]
    roots.push(home.join(".local/share/applications"));
    #[cfg(windows)]
    roots.push(home.join("AppData").join("Local").join("Programs"));
    roots
}

fn path_string_contains_notepad(p: &Path) -> bool {
    p.to_string_lossy().to_lowercase().contains("notepad")
}

fn looks_like_shared_lib(p: &Path) -> bool {
    matches!(
        p.extension().and_then(|e| e.to_str()).map(|e| e.to_lowercase()),
        Some(ext) if ext == "so" || ext == "dylib" || ext == "dll"
    )
}

#[cfg(unix)]
fn is_user_executable_file(p: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    if !p.is_file() || looks_like_shared_lib(p) {
        return false;
    }
    fs::metadata(p)
        .map(|m| m.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

#[cfg(windows)]
fn is_user_executable_file(p: &Path) -> bool {
    if !p.is_file() || looks_like_shared_lib(p) {
        return false;
    }
    let ext = p
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());
    matches!(
        ext.as_deref(),
        Some("exe") | Some("bat") | Some("cmd") | Some("appimage")
    ) || ext.is_none()
}

#[cfg(not(any(unix, windows)))]
fn is_user_executable_file(p: &Path) -> bool {
    p.is_file() && !looks_like_shared_lib(p)
}

fn parse_desktop_exec_binary(desktop: &Path) -> Option<PathBuf> {
    let text = fs::read_to_string(desktop).ok()?;
    let line = text.lines().find(|l| l.starts_with("Exec="))?;
    let rest = line.strip_prefix("Exec=")?.trim();
    for token in rest.split_whitespace() {
        if token == "env" || token.starts_with('-') || token.starts_with('%') {
            continue;
        }
        if token.contains('=') && !token.starts_with('/') {
            continue;
        }
        let t = token.trim_matches('"');
        if t.starts_with('/') {
            let pb = PathBuf::from(t);
            if pb.is_file() {
                return Some(pb);
            }
        }
    }
    None
}

fn find_notepad_under_user_app_folders() -> Option<PathBuf> {
    let mut hits: Vec<PathBuf> = Vec::new();

    for root in user_notepad_search_roots() {
        if !root.is_dir() {
            continue;
        }
        if root.ends_with(Path::new("applications")) {
            let Ok(rd) = fs::read_dir(&root) else {
                continue;
            };
            for e in rd.flatten() {
                let p = e.path();
                if !p.is_file() {
                    continue;
                }
                let Some(fname) = p.file_name().and_then(|n| n.to_str()) else {
                    continue;
                };
                let name = fname.to_lowercase();
                if !name.ends_with(".desktop") || !name.contains("notepad") {
                    continue;
                }
                if let Some(bin) = parse_desktop_exec_binary(&p) {
                    hits.push(bin);
                }
            }
            continue;
        }

        for entry in WalkDir::new(&root).max_depth(6).into_iter().filter_map(|e| e.ok()) {
            let p = entry.path();
            if !path_string_contains_notepad(p) {
                continue;
            }
            if p.is_file() && is_user_executable_file(p) {
                hits.push(p.to_path_buf());
            }
        }
    }

    hits.sort();
    hits.dedup();
    if let Some(p) = hits.iter().find(|p| {
        p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
            n.eq_ignore_ascii_case("notepad") || n.eq_ignore_ascii_case("notepad.exe")
        })
    }) {
        return Some(p.clone());
    }
    hits.into_iter().next()
}

fn sessions_root_dir() -> PathBuf {
    if let Some(home) = dirs::home_dir() {
        home.join(".opencage").join("sessions")
    } else {
        PathBuf::from(".opencage/sessions")
    }
}

fn session_file_path(session_id: &str) -> PathBuf {
    sessions_root_dir().join(format!("{session_id}.json"))
}

/// Map the user-facing Anthropic model names to API model ids.
fn anthropic_model_id(short: &str) -> Option<&'static str> {
    // All three 4.7 tiers are selectable. Opus 4.7 is live; Sonnet/Haiku 4.7 may not be yet, in
    // which case the provider transparently falls back to the latest tier model (4.6 / 4.5).
    match short.trim().to_lowercase().as_str() {
        "opus-4.7" | "opus" | "claude-opus-4-7" => Some("claude-opus-4-7"),
        "sonnet-4.7" | "sonnet" | "claude-sonnet-4-7" => Some("claude-sonnet-4-7"),
        "sonnet-4.6" | "claude-sonnet-4-6" => Some("claude-sonnet-4-6"),
        "haiku-4.7" | "haiku" | "claude-haiku-4-7" => Some("claude-haiku-4-7"),
        "haiku-4.5" | "claude-haiku-4-5-20251001" => Some("claude-haiku-4-5-20251001"),
        _ => None,
    }
}

/// Open a URL in the user's default browser (Linux openers, with a macOS fallback).
fn open_in_browser(url: &str) {
    for opener in ["xdg-open", "sensible-browser", "x-www-browser", "open"] {
        if Command::new(opener).arg(url).spawn().is_ok() {
            return;
        }
    }
}

fn delete_session_file(session_id: &str) -> Result<()> {
    let path = session_file_path(session_id);
    if path.exists() {
        fs::remove_file(&path)
            .with_context(|| format!("Failed to delete {}", path.display()))?;
    }
    Ok(())
}

fn generate_session_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{:x}", nanos)
}

fn save_session_messages(session_id: &str, messages: &[Message]) -> Result<()> {
    fs::create_dir_all(sessions_root_dir())?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let file = SessionFile {
        title: summarize_session_title(messages),
        updated_ts: now,
        messages: messages.to_vec(),
    };
    let text = serde_json::to_string(&file)?;
    fs::write(session_file_path(session_id), text)?;
    Ok(())
}

/// Write an imported conversation as a session file, preserving its source timestamp.
fn save_imported_session(session_id: &str, messages: &[Message], updated_ts: u64) -> Result<()> {
    fs::create_dir_all(sessions_root_dir())?;
    let file = SessionFile {
        title: summarize_session_title(messages),
        updated_ts,
        messages: messages.to_vec(),
    };
    fs::write(session_file_path(session_id), serde_json::to_string(&file)?)?;
    Ok(())
}

fn load_session_messages(session_id: &str) -> Result<Vec<Message>> {
    let path = session_file_path(session_id);
    if !path.exists() {
        return Ok(vec![]);
    }
    let text = fs::read_to_string(path)?;
    if let Ok(file) = serde_json::from_str::<SessionFile>(&text) {
        return Ok(file.messages);
    }
    let msgs: Vec<Message> = serde_json::from_str(&text)?;
    Ok(msgs)
}

fn list_sessions() -> Vec<SessionMeta> {
    let root = sessions_root_dir();
    let mut out: Vec<SessionMeta> = vec![];
    let Ok(entries) = fs::read_dir(root) else {
        return out;
    };
    for e in entries.filter_map(|x| x.ok()) {
        let p = e.path();
        if p.extension().and_then(|x| x.to_str()) != Some("json") {
            continue;
        }
        let Some(stem) = p.file_stem().and_then(|x| x.to_str()) else {
            continue;
        };
        let text = fs::read_to_string(&p).unwrap_or_default();
        if let Ok(file) = serde_json::from_str::<SessionFile>(&text) {
            out.push(SessionMeta {
                id: stem.to_string(),
                title: file.title,
                updated_ts: file.updated_ts,
            });
        } else if let Ok(messages) = serde_json::from_str::<Vec<Message>>(&text) {
            out.push(SessionMeta {
                id: stem.to_string(),
                title: summarize_session_title(&messages),
                updated_ts: 0,
            });
        }
    }
    out.sort_by_key(|s| std::cmp::Reverse(s.updated_ts));
    out
}

fn summarize_session_title(messages: &[Message]) -> String {
    let user_msg = messages
        .iter()
        .find(|m| m.role == "user")
        .map(|m| m.content.trim())
        .unwrap_or("New session");
    let mut clean = user_msg
        .replace('\n', " ")
        .split_whitespace()
        .take(10)
        .collect::<Vec<_>>()
        .join(" ");
    if clean.is_empty() {
        clean = "New session".to_string();
    }
    if user_msg.split_whitespace().count() > 10 {
        clean.push_str("...");
    }
    clean
}

#[derive(Debug, Deserialize)]
struct AutocodePlan {
    summary: Option<String>,
    files: Option<Vec<AutocodeFile>>,
    commands: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct AutocodeFile {
    path: String,
    content: String,
}

pub fn is_coding_task(prompt: &str) -> bool {
    let p = prompt.to_lowercase();
    ["implement", "code", "create", "build", "refactor", "fix", "edit", "update"]
        .iter()
        .any(|k| p.contains(k))
}

pub fn run_autonomous_coding(
    settings: &Settings,
    history: &[Message],
    prompt: &str,
    deep_think_enabled: bool,
    deep_think_level: u8,
    coding_expanded: bool,
    cwd: &Path,
) -> Result<AgentResult> {
    let raw = sanitize_model_text(&query_coding_actions(
        settings,
        history,
        prompt,
        deep_think_enabled,
        deep_think_level,
        coding_expanded,
    )?);
    let plan = match parse_autocode_plan(&raw) {
        Ok(p) => p,
        Err(_) => {
            let retry_prompt = format!("{prompt}\n\nReply with OPENCAGE_FILE/OPENCAGE_CMD blocks FIRST (no markdown fences) so actions can be executed.");
            let retry_raw = sanitize_model_text(&query_coding_actions(
                settings,
                history,
                &retry_prompt,
                deep_think_enabled,
                deep_think_level,
                coding_expanded,
            )?);
            parse_autocode_plan(&retry_raw).with_context(|| {
                format!(
                    "Failed to parse coding action output. Raw sample: {}",
                    truncate_for_error(&retry_raw, 240)
                )
            })?
        }
    };

    let mut written_files: Vec<String> = Vec::new();
    let mut skipped_files: Vec<String> = Vec::new();
    let mut command_plan = Vec::new();
    if let Some(files) = plan.files {
        for f in files {
            if let Some(full) = safe_join_under(cwd, &f.path) {
                if let Some(parent) = full.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::write(&full, f.content)?;
                let rel = full
                    .strip_prefix(cwd)
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|_| full.display().to_string());
                written_files.push(rel);
            } else {
                skipped_files.push(f.path);
            }
        }
    }

    if let Some(cmds) = plan.commands {
        for cmd in cmds.into_iter().take(8) {
            command_plan.push(cmd);
        }
    }

    let mut sections: Vec<String> = Vec::new();
    sections.push(
        plan.summary
            .unwrap_or_else(|| "Autocode completed.".to_string()),
    );
    if !written_files.is_empty() {
        let listed = written_files
            .iter()
            .map(|f| format!("  ✅ {f}"))
            .collect::<Vec<_>>()
            .join("\n");
        sections.push(format!("📁 Files created/updated:\n{listed}"));
    } else {
        sections.push("📁 No files were created.".to_string());
    }
    if !skipped_files.is_empty() {
        let listed = skipped_files
            .iter()
            .map(|f| format!("  ⚠️ {f}"))
            .collect::<Vec<_>>()
            .join("\n");
        sections.push(format!("Skipped unsafe paths:\n{listed}"));
    }
    if !command_plan.is_empty() {
        sections.push("💬 Commands queued — approve each with yes/no.".to_string());
    }

    Ok(AgentResult {
        text: sections.join("\n\n"),
        commands: command_plan,
    })
}

fn safe_join_under(root: &Path, rel: &str) -> Option<PathBuf> {
    let candidate = root.join(rel);
    let cleaned = candidate.components().fold(PathBuf::new(), |mut acc, c| {
        if matches!(c, std::path::Component::ParentDir) {
            return acc;
        }
        acc.push(c.as_os_str());
        acc
    });
    if cleaned.starts_with(root) {
        Some(cleaned)
    } else {
        None
    }
}

fn extract_json_object(raw: &str) -> Option<String> {
    let start = raw.find('{')?;
    let end = raw.rfind('}')?;
    if end <= start {
        return None;
    }
    Some(raw[start..=end].to_string())
}

/// Drop control characters that never belong in code/text (keeps newlines and tabs), so a
/// stray byte from the model can't corrupt a generated file or the displayed summary.
fn sanitize_model_text(s: &str) -> String {
    s.chars()
        .filter(|c| !c.is_control() || matches!(c, '\n' | '\r' | '\t'))
        .collect()
}

fn parse_autocode_plan(raw: &str) -> Result<AutocodePlan> {
    if let Some(plan) = parse_tagged_autocode(raw) {
        return Ok(plan);
    }
    if let Ok(plan) = serde_json::from_str::<AutocodePlan>(raw) {
        return Ok(plan);
    }
    if let Some(json) = extract_code_fenced_json(raw)
        && let Ok(plan) = serde_json::from_str::<AutocodePlan>(&json)
    {
        return Ok(plan);
    }
    if let Some(json) = extract_json_object(raw)
        && let Ok(plan) = serde_json::from_str::<AutocodePlan>(&json)
    {
        return Ok(plan);
    }
    Err(anyhow::anyhow!("Failed to parse coding action output"))
}

fn parse_tagged_autocode(raw: &str) -> Option<AutocodePlan> {
    let mut files = Vec::new();
    let mut cmds = Vec::new();

    let mut rest = raw;
    let mut truncated = false;
    while let Some(start) = rest.find("<OPENCAGE_FILE path=\"") {
        let after = &rest[start + "<OPENCAGE_FILE path=\"".len()..];
        let Some(end_quote) = after.find('"') else {
            break;
        };
        let path = after[..end_quote].to_string();
        let after_tag = &after[end_quote..];
        let Some(open_end) = after_tag.find('>') else {
            break;
        };
        let content_start = &after_tag[open_end + 1..];
        match content_start.find("</OPENCAGE_FILE>") {
            Some(close) => {
                let content = content_start[..close].trim_matches('\n').to_string();
                files.push(AutocodeFile { path, content });
                rest = &content_start[close + "</OPENCAGE_FILE>".len()..];
            }
            None => {
                // Response was cut off (token limit) before the closing tag — salvage the
                // partial file so the work isn't lost, and flag it.
                let content = content_start.trim_matches('\n').to_string();
                if !content.trim().is_empty() {
                    files.push(AutocodeFile { path, content });
                    truncated = true;
                }
                break;
            }
        }
    }

    let mut rest_cmd = raw;
    while let Some(start) = rest_cmd.find("<OPENCAGE_CMD>") {
        let after = &rest_cmd[start + "<OPENCAGE_CMD>".len()..];
        if let Some(end) = after.find("</OPENCAGE_CMD>") {
            let cmd = after[..end].trim().to_string();
            if !cmd.is_empty() {
                cmds.push(cmd);
            }
            rest_cmd = &after[end + "</OPENCAGE_CMD>".len()..];
        } else {
            break;
        }
    }

    let mut summary = raw
        .split("<OPENCAGE_FILE")
        .next()
        .unwrap_or(raw)
        .split("<OPENCAGE_CMD>")
        .next()
        .unwrap_or(raw)
        .trim()
        .to_string();
    if truncated {
        if !summary.is_empty() {
            summary.push_str("\n\n");
        }
        summary.push_str(
            "⚠ Output was truncated before completion — the last file may be incomplete. Ask me to continue it.",
        );
    }

    if files.is_empty() && cmds.is_empty() {
        None
    } else {
        Some(AutocodePlan {
            summary: if summary.is_empty() { None } else { Some(summary) },
            files: Some(files),
            commands: Some(cmds),
        })
    }
}

fn extract_code_fenced_json(raw: &str) -> Option<String> {
    let start = raw.find("```")?;
    let rest = &raw[start + 3..];
    let rest = if let Some(stripped) = rest.strip_prefix("json") {
        stripped
    } else {
        rest
    };
    let rest = rest.strip_prefix('\n').unwrap_or(rest);
    let end = rest.find("```")?;
    Some(rest[..end].trim().to_string())
}

fn truncate_for_error(raw: &str, max: usize) -> String {
    let s = raw.replace('\n', " ");
    if s.len() <= max {
        s
    } else {
        format!("{}...", &s[..max])
    }
}

fn parse_yes_no(raw: &str) -> Option<bool> {
    match raw.trim().to_lowercase().as_str() {
        "y" | "yes" => Some(true),
        "n" | "no" => Some(false),
        _ => None,
    }
}

fn parse_agent_tag(line: &str) -> Option<&str> {
    let start = line.find("[sub-agent:")?;
    let rest = &line[start + 11..];
    let end = rest.find(']')?;
    Some(&rest[..end])
}
