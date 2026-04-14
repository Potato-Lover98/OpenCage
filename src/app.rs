use std::collections::VecDeque;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::{Duration, Instant};
use std::time::SystemTime;

use anyhow::{Context, Result};
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyModifiers,
    MouseButton, MouseEvent, MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    size,
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use serde::Deserialize;
use walkdir::WalkDir;

use crate::ai::providers::{
    query_coding_actions, query_with_subagent, select_subagents, validate_settings_keys,
};
use crate::ai::sandbox::run_in_sandbox;
use crate::ai::rag::RagStore;
use crate::core::config::SettingsStore;
use crate::core::models::{FileNode, Message, Provider, Settings};
use crate::ui;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveTab {
    Chat,
    Settings,
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
    pub copilot: String,
    pub report: Vec<String>,
}

pub struct App {
    pub settings: Settings,
    pub input: String,
    pub messages: Vec<Message>,
    pub files: Vec<FileNode>,
    pub selected_file_idx: usize,
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
    expanded_paths: HashSet<PathBuf>,
    pub context_window_limit_chars: usize,
    pending_command: Option<String>,
    queued_commands: VecDeque<String>,
    pending_trust_prompt: bool,
    deferred_coding_prompt: Option<String>,
    rag: RagStore,
    blacklist_last_modified: Option<SystemTime>,
}

pub struct AgentResult {
    pub text: String,
    pub commands: Vec<String>,
}

impl App {
    pub fn new(settings: Settings, rag: RagStore) -> Self {
        let current_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let current_dir_key = current_dir.to_string_lossy().to_string();
        let trusted = settings.trusted_paths.contains(&current_dir_key);
        let mut app = Self {
            settings,
            input: String::new(),
            messages: vec![Message {
                role: "system".to_string(),
                content: "Welcome to Opencage. Type / to open command palette.".to_string(),
            }],
            files: vec![],
            selected_file_idx: 0,
            show_palette: false,
            palette_commands: vec![
                "/settings",
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
            expanded_paths: HashSet::new(),
            context_window_limit_chars: 120_000,
            pending_command: None,
            queued_commands: VecDeque::new(),
            pending_trust_prompt: false,
            deferred_coding_prompt: None,
            rag,
            blacklist_last_modified: None,
        };
        if !trusted {
            app.messages.push(Message {
                role: "system".to_string(),
                content: "This folder is not trusted yet. When coding is requested, I will ask yes/no before trusting it."
                    .to_string(),
            });
        }
        app.refresh_file_tree();
        app
    }

    pub fn run(&mut self) -> Result<()> {
        enable_raw_mode()?;
        let mut stdout = std::io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = ratatui::backend::CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;
        let tick = Duration::from_millis(120);
        let mut should_quit = false;

        while !should_quit {
            self.sync_cwd_and_tree();
            self.sync_blacklist_file();
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
                        let _ = self.rag.remember("assistant", &final_reply);
                        for line in final_reply.lines() {
                            if let Some(agent_name) = parse_agent_tag(line) {
                                let scope = format!("agent:{agent_name}");
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

        disable_raw_mode()?;
        execute!(terminal.backend_mut(), DisableMouseCapture, LeaveAlternateScreen)?;
        terminal.show_cursor()?;
        Ok(())
    }

    fn handle_key(&mut self, key: KeyEvent) -> bool {
        if self.active_tab == ActiveTab::Settings {
            return self.handle_settings_key(key);
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
                KeyCode::Enter => {
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
                code: KeyCode::Char('c'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => return true,
            KeyEvent {
                code: KeyCode::Tab, ..
            } => self.toggle_node(),
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
                code: KeyCode::PageUp,
                ..
            } => {
                self.chat_scroll = self.chat_scroll.saturating_add(3);
            }
            KeyEvent {
                code: KeyCode::PageDown,
                ..
            } => {
                self.chat_scroll = self.chat_scroll.saturating_sub(3);
            }
            KeyEvent {
                code: KeyCode::Up, ..
            } => self.selected_file_idx = self.selected_file_idx.saturating_sub(1),
            KeyEvent {
                code: KeyCode::Down,
                ..
            } => self.selected_file_idx = (self.selected_file_idx + 1).min(self.files.len() - 1),
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
                form.selected = (form.selected + 1).min(8);
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
                code: KeyCode::Enter,
                ..
            } if form.selected == 3 => {
                form.cuss_filter = !form.cuss_filter;
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
                if matches!(form.selected, 2 | 4 | 5 | 6 | 7 | 8) {
                    Self::form_field_mut(form).pop();
                }
            }
            KeyEvent {
                code: KeyCode::Char(ch),
                ..
            } => {
                if matches!(form.selected, 2 | 4 | 5 | 6 | 7 | 8) {
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
            8 => &mut form.copilot,
            _ => &mut form.avatar,
        }
    }

    fn handle_mouse(&mut self, mouse: MouseEvent) {
        if mouse.kind != MouseEventKind::Down(MouseButton::Left) {
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
        let width = size().map(|(w, _)| w).unwrap_or(120);
        let left_width = width.saturating_mul(35) / 100;
        if mouse.column >= left_width || mouse.row == 0 {
            return;
        }
        let idx = mouse.row.saturating_sub(1) as usize;
        if idx < self.files.len() {
            self.selected_file_idx = idx;
            if self.files[idx].is_dir && self.is_arrow_click(idx, mouse.column) {
                self.toggle_node();
            }
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
                Some(true) => {
                    if let Some(cmd) = self.pending_command.take() {
                        let out = run_in_sandbox(&cmd, &self.settings, true)
                            .unwrap_or_else(|e| format!("Command failed: {e}"));
                        self.messages.push(Message {
                            role: "system".to_string(),
                            content: format!("Executed: {cmd}\n{out}"),
                        });
                    }
                    self.prompt_next_command_if_needed();
                }
                Some(false) => {
                    if let Some(cmd) = self.pending_command.take() {
                        self.messages.push(Message {
                            role: "system".to_string(),
                            content: format!("Denied command: {cmd}"),
                        });
                    }
                    self.prompt_next_command_if_needed();
                }
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
            self.messages.push(Message {
                role: "system".to_string(),
                content: "Queued command from input. Awaiting approval.".to_string(),
            });
            return;
        }

        let user_text = self.filter_cuss(raw);
        self.messages.push(Message {
            role: "user".to_string(),
            content: user_text.clone(),
        });
        let _ = self.rag.remember("user", &user_text);

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
            let scope = format!("agent:{}", agent.name());
            let _ = self.rag.remember_scoped(&scope, "user", &user_text);
            let ctx = self
                .rag
                .retrieve_scoped(&scope, &user_text, 3)
                .unwrap_or_default();
            rag_per_agent.push((scope, ctx));
        }
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
            "/settings" => {
                self.open_settings_tab();
            }
            "/help" => {
                self.messages.push(Message {
                    role: "system".to_string(),
                    content: "Commands: /settings /avatar /model /buddy /btw /blacklist /clear /refresh /remember /memories /deep(on|off|0-10) /expand(on|off)"
                        .to_string(),
                });
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
            "/model" => {
                self.messages.push(Message {
                    role: "system".to_string(),
                    content: format!(
                        "Current provider={} model={}",
                        self.settings.provider.as_str(),
                        self.settings.model
                    ),
                });
            }
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
                        Ok(path) => self.messages.push(Message {
                            role: "system".to_string(),
                            content: format!(
                                "Opened blacklist file in editor:\n{}",
                                path.display()
                            ),
                        }),
                        Err(e) => self.messages.push(Message {
                            role: "system".to_string(),
                            content: format!("Could not open blacklist file: {e}"),
                        }),
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
                    let _ = self.rag.remember_scoped("global", "user", &text);
                    self.messages.push(Message {
                        role: "system".to_string(),
                        content: "Saved to long-term memory.".to_string(),
                    });
                }
            }
            "/memories" => {
                let q = parts.collect::<Vec<_>>().join(" ");
                let query = if q.is_empty() { "user" } else { &q };
                let items = self.rag.retrieve_scoped("global", query, 5).unwrap_or_default();
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
            _ => self.messages.push(Message {
                role: "system".to_string(),
                content: "Unknown command".to_string(),
            }),
        }
    }

    fn open_blacklist_editor(&self) -> Result<PathBuf> {
        let mut commands: Vec<_> = self.settings.blocked_commands.iter().cloned().collect();
        commands.sort();
        let default_preconfigured = ["rm", "shutdown", "reboot", "mkfs", "dd"];
        let mut lines = vec![
            "# Opencage Blacklisted Commands".to_string(),
            "# One command per line".to_string(),
            "# Lines starting with # are comments".to_string(),
            String::new(),
            "# Preconfigured defaults (uncomment to enable)".to_string(),
        ];
        for c in default_preconfigured {
            lines.push(format!("# {c}"));
        }
        lines.push(String::new());
        lines.push("# Current active blacklist".to_string());
        lines.extend(commands);
        lines.push(String::new());

        let config_dir = dirs::config_dir()
            .context("Failed to resolve config directory")?
            .join("opencage");
        fs::create_dir_all(&config_dir)?;
        let path = config_dir.join("blacklist.txt");
        fs::write(&path, lines.join("\n"))?;

        let path_str = path.to_string_lossy().to_string();
        if let Ok(editor) = std::env::var("EDITOR") {
            let _ = Command::new(editor).arg(&path_str).spawn();
            return Ok(path);
        }
        if Command::new("xdg-open").arg(&path_str).spawn().is_ok() {
            return Ok(path);
        }
        for editor in ["gedit", "kate", "mousepad", "leafpad", "nano", "vi"] {
            if Command::new(editor).arg(&path_str).spawn().is_ok() {
                return Ok(path);
            }
        }
        Err(anyhow::anyhow!(
            "No editor found. Set $EDITOR or install xdg-open/gedit."
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
        let mut nodes = vec![FileNode {
            path: root.clone(),
            depth: 0,
            is_dir: true,
            expanded: true,
        }];
        self.collect_nodes(&root, 1, &mut nodes);
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
            self.pending_command = Some(cmd.clone());
            self.messages.push(Message {
                role: "system".to_string(),
                content: format!("Run this command?\n{cmd}\nType yes/y or no/n."),
            });
        }
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
            copilot: self.settings.github_copilot_token.clone().unwrap_or_default(),
            report: vec!["Press Ctrl+S to save, F5 to validate, Esc to close".to_string()],
        });
        self.active_tab = ActiveTab::Settings;
        self.last_status = "Settings tab opened".to_string();
    }

    fn settings_form_to_settings(&self, form: &SettingsForm) -> Settings {
        let mut s = self.settings.clone();
        s.provider = Provider::all()[form.provider_idx].clone();
        s.model = s.provider.models()[form.model_idx].to_string();
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
        self.palette_commands
            .iter()
            .copied()
            .filter(|cmd| cmd.starts_with(&self.input))
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
        self.pending_command
            .as_ref()
            .map(|cmd| format!("Run command?\n{cmd}\nType yes/y or no/n"))
    }

    fn is_arrow_click(&self, idx: usize, column: u16) -> bool {
        let Some(node) = self.files.get(idx) else {
            return false;
        };
        // Left border (1) + marker "> " (2) + indentation.
        let arrow_col = 1 + 2 + (2 * node.depth.saturating_sub(1)) as u16;
        column >= arrow_col.saturating_sub(1) && column <= arrow_col + 1
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
        let input_h = 3usize;
        let start = lines.saturating_sub(input_h + panel_h);
        let end = start + panel_h.saturating_sub(2);
        Some((start, end))
    }
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

fn is_coding_task(prompt: &str) -> bool {
    let p = prompt.to_lowercase();
    ["implement", "code", "create", "build", "refactor", "fix", "edit", "update"]
        .iter()
        .any(|k| p.contains(k))
}

fn run_autonomous_coding(
    settings: &Settings,
    history: &[Message],
    prompt: &str,
    deep_think_enabled: bool,
    deep_think_level: u8,
    coding_expanded: bool,
    cwd: &Path,
) -> Result<AgentResult> {
    let raw = query_coding_actions(
        settings,
        history,
        prompt,
        deep_think_enabled,
        deep_think_level,
        coding_expanded,
    )?;
    let plan = match parse_autocode_plan(&raw) {
        Ok(p) => p,
        Err(_) => {
            let retry_prompt = format!("{prompt}\n\nPlease include OPENCAGE_FILE/OPENCAGE_CMD blocks so actions can be executed.");
            let retry_raw = query_coding_actions(
                settings,
                history,
                &retry_prompt,
                deep_think_enabled,
                deep_think_level,
                coding_expanded,
            )?;
            parse_autocode_plan(&retry_raw).with_context(|| {
                format!(
                    "Failed to parse coding action output. Raw sample: {}",
                    truncate_for_error(&retry_raw, 240)
                )
            })?
        }
    };

    let mut report = Vec::new();
    let mut command_plan = Vec::new();
    if let Some(files) = plan.files {
        for f in files {
            if let Some(full) = safe_join_under(cwd, &f.path) {
                if let Some(parent) = full.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::write(&full, f.content)?;
                report.push(format!("wrote {}", full.display()));
            } else {
                report.push(format!("skipped unsafe path {}", f.path));
            }
        }
    }

    if let Some(cmds) = plan.commands {
        for cmd in cmds.into_iter().take(8) {
            command_plan.push(cmd);
        }
        report.push("Commands queued for per-command yes/no confirmation.".to_string());
    }

    report.insert(
        0,
        plan.summary
            .unwrap_or_else(|| "Autocode completed.".to_string()),
    );
    Ok(AgentResult {
        text: report.join("\n\n"),
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
        let Some(close) = content_start.find("</OPENCAGE_FILE>") else {
            break;
        };
        let content = content_start[..close].trim_matches('\n').to_string();
        files.push(AutocodeFile { path, content });
        rest = &content_start[close + "</OPENCAGE_FILE>".len()..];
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

    let summary = raw
        .split("<OPENCAGE_FILE")
        .next()
        .unwrap_or(raw)
        .split("<OPENCAGE_CMD>")
        .next()
        .unwrap_or(raw)
        .trim()
        .to_string();

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
