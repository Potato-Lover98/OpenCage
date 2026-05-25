use std::path::Path;

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::prelude::*;
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap};

use crate::app::{ActiveTab, AlertKind, App};
use crate::core::models::{Message, Provider};

/// Bottom strip height (input + deep think). Keep in sync everywhere this layout is duplicated.
pub const CHAT_FOOTER_HEIGHT: u16 = 8;

/// Main chat split: file tree, chat messages area, optional command bar + palette + footer.
pub struct ChatMainLayout {
    pub files: Rect,
    pub command_bar: Option<Rect>,
    pub chat: Rect,
    pub palette: Rect,
    pub footer: Rect,
}

pub fn chat_main_layout(area: Rect, show_palette: bool, has_command_approval: bool) -> ChatMainLayout {
    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(1),
            Constraint::Length(if show_palette { 6 } else { 0 }),
            Constraint::Length(CHAT_FOOTER_HEIGHT),
        ])
        .split(area);
    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(35), Constraint::Percentage(65)])
        .split(root[0]);
    let chat_column = body[1];
    let (command_bar, chat) = if has_command_approval {
        let split = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(1)])
            .split(chat_column);
        (Some(split[0]), split[1])
    } else {
        (None, chat_column)
    };
    ChatMainLayout {
        files: body[0],
        command_bar,
        chat,
        palette: root[1],
        footer: root[2],
    }
}

fn file_tree_title(current_dir: &Path) -> String {
    let label = current_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("/");
    format!("🌳 {label} [Tab · click folder]")
}

/// Inner area of the file tree list (inside borders + title row).
pub fn file_tree_content_rect(files_panel: Rect, current_dir: &Path) -> Rect {
    Block::default()
        .title(file_tree_title(current_dir))
        .borders(Borders::ALL)
        .inner(files_panel)
}

fn chat_panel_block(app: &App) -> Block<'_> {
    Block::default()
        .title(format!(
            "💫 {} [{}:{}] · Pg↑↓ Ctrl+↑↓ · wheel",
            app.settings.ai_avatar,
            app.settings.provider.as_str(),
            app.settings.model
        ))
        .borders(Borders::ALL)
}

/// Inner text area of the chat panel (must match [`draw`]).
pub fn chat_message_inner_rect(chat_panel: Rect, app: &App) -> Rect {
    chat_panel_block(app).inner(chat_panel)
}

/// One message may contain `\n`; split into multiple [`Line`]s so the chat shows real line breaks.
fn append_message_as_chat_lines(all_lines: &mut Vec<Line>, m: &Message) {
    let (emoji, style) = match m.role.as_str() {
        "user" => ("🧑", Style::default().fg(Color::LightGreen)),
        "assistant" => ("🤖", Style::default().fg(Color::LightBlue)),
        "system" => ("🛟", Style::default().fg(Color::Magenta)),
        _ => ("💬", Style::default().fg(Color::White)),
    };
    let head = style.add_modifier(Modifier::BOLD);
    let parts: Vec<&str> = if m.content.is_empty() {
        vec![""]
    } else {
        m.content.split('\n').collect()
    };
    for (i, part) in parts.iter().enumerate() {
        if i == 0 {
            all_lines.push(Line::from(vec![
                Span::styled(format!("{emoji} "), head),
                Span::raw((*part).to_string()),
            ]));
        } else {
            all_lines.push(Line::from(vec![
                Span::raw("    "),
                Span::raw((*part).to_string()),
            ]));
        }
    }
}

/// Approximate wrapped line count (word wrap at `text_width`) for Paragraph scroll math.
fn count_wrapped_chat_lines(lines: &[Line], text_width: u16) -> usize {
    let w = text_width.max(1) as usize;
    let mut total = 0usize;
    for line in lines {
        let s = line.to_string();
        if s.is_empty() {
            total += 1;
        } else {
            total += textwrap::wrap(&s, w).len();
        }
    }
    total.max(1)
}

pub fn draw(frame: &mut Frame, app: &App) {
    if app.active_tab == ActiveTab::Settings {
        draw_settings(frame, app);
        return;
    }
    if app.active_tab == ActiveTab::Alerts {
        draw_alerts(frame, app);
        return;
    }
    let layout = chat_main_layout(
        frame.area(),
        app.show_palette,
        app.command_approval_bar().is_some(),
    );

    let items: Vec<ListItem> = app
        .files
        .iter()
        .enumerate()
        .map(|(idx, n)| {
            let name = if n.depth == 0 {
                app.current_dir
                    .file_name()
                    .and_then(|x| x.to_str())
                    .unwrap_or("/")
                    .to_string()
            } else {
                n.path
                    .file_name()
                    .and_then(|x| x.to_str())
                    .unwrap_or(".")
                    .to_string()
            };
            let prefix = if n.is_dir {
                if n.expanded { "v" } else { ">" }
            } else {
                " "
            };
            let icon = if n.is_dir { "📁" } else { "📄" };
            let indent = "  ".repeat(n.depth.saturating_sub(1));
            let row_selected =
                app.file_tree_highlight && idx == app.selected_file_idx;
            let marker = if row_selected { "> " } else { "  " };
            let style = if row_selected {
                Style::default().fg(Color::Black).bg(Color::Cyan).add_modifier(Modifier::BOLD)
            } else if n.is_dir {
                Style::default().fg(Color::Yellow)
            } else {
                Style::default().fg(Color::Green)
            };
            ListItem::new(Line::styled(
                format!("{marker}{indent}{prefix} {icon} {name}"),
                style,
            ))
        })
        .collect();
    let file_list = List::new(items).block(
        Block::default()
            .title(file_tree_title(&app.current_dir))
            .borders(Borders::ALL),
    );
    frame.render_widget(file_list, layout.files);

    let mut all_lines: Vec<Line> = vec![];
    if app.is_push_to_talk_active() {
        all_lines.push(voice_streaming_line(app.voice_meter_level()));
        all_lines.push(Line::from(""));
    }
    for m in &app.messages {
        append_message_as_chat_lines(&mut all_lines, m);
    }
    let chat_area = layout.chat;
    if let (Some(bar_area), Some((cmd, yes_selected))) =
        (layout.command_bar, app.command_approval_bar())
    {
        let yes_style = if yes_selected {
            Style::default().fg(Color::Black).bg(Color::Green)
        } else {
            Style::default().fg(Color::Green)
        };
        let no_style = if yes_selected {
            Style::default().fg(Color::Red)
        } else {
            Style::default().fg(Color::Black).bg(Color::Red)
        };
        let bar = Paragraph::new(Line::from(vec![
            Span::raw("Run command: "),
            Span::styled(cmd, Style::default().fg(Color::Yellow)),
            Span::raw("   "),
            Span::styled(" Yes ", yes_style),
            Span::raw(" "),
            Span::styled(" No ", no_style),
            Span::raw("   (Left/Right + Enter)"),
        ]))
        .block(Block::default().borders(Borders::ALL).title("⚠ Command Approval"));
        frame.render_widget(bar, bar_area);
    }

    let chat_block = chat_panel_block(app);
    let chat_inner = chat_block.inner(chat_area);
    let text_cols = chat_inner.width.max(1);
    let text_rows = chat_inner.height as usize;
    let total_wrapped = count_wrapped_chat_lines(&all_lines, text_cols);
    let max_from_bottom = total_wrapped.saturating_sub(text_rows);
    let clipped_scroll = app.chat_scroll.min(max_from_bottom);
    let scroll_y = max_from_bottom.saturating_sub(clipped_scroll);
    let scroll_y = scroll_y.min(u16::MAX as usize) as u16;
    let chat = Paragraph::new(all_lines)
        .block(chat_block)
        .wrap(Wrap { trim: true })
        .scroll((scroll_y, 0));
    frame.render_widget(chat, chat_area);

    if app.show_palette {
        let filtered = app.filtered_suggestions();
        let cmd_items: Vec<ListItem> = filtered
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let marker = if i == app.palette_selected { "▶" } else { " " };
                let style = if i == app.palette_selected {
                    Style::default().fg(Color::Black).bg(Color::LightMagenta)
                } else {
                    Style::default().fg(Color::Cyan)
                };
                ListItem::new(Line::styled(format!("{marker} {c}"), style))
            })
            .collect();
        let list = List::new(cmd_items).block(
            Block::default()
                .title("✨ Slash Suggestions (↑/↓ + Enter or click)")
                .borders(Borders::ALL),
        );
        let mut state = ListState::default();
        if !filtered.is_empty() {
            state.select(Some(app.palette_selected.min(filtered.len() - 1)));
        }
        frame.render_stateful_widget(list, layout.palette, &mut state);
    }

    let footer = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(layout.footer);

    let input_text: Text = if app.voice_partial.is_empty() {
        Text::raw(app.input.clone())
    } else {
        Text::from(Line::from(vec![
            Span::raw(app.input.clone()),
            Span::styled(
                format!(" {}", app.voice_partial),
                Style::default().fg(Color::Yellow),
            ),
        ]))
    };
    let input_block = Block::default()
        .title(format!(
            "⌨️ Input | {} | newline: Alt+Enter or Ctrl+J · Shift+Enter (CSI-u terminals)",
            app.last_status
        ))
        .borders(Borders::ALL);
    let input = Paragraph::new(input_text)
        .block(input_block)
        .wrap(Wrap { trim: true });
    frame.render_widget(input, footer[0]);

    let deep = app.deep_think_level.min(10);
    let filled = "#".repeat(deep as usize);
    let empty = "-".repeat((10 - deep) as usize);
    let status = if app.deep_think_enabled { "ON" } else { "OFF" };
    let coding_mode = if app.coding_expanded { "expanded" } else { "compact" };
    let used = app.context_usage_chars();
    let limit = app.context_window_limit_chars.max(1);
    let pct = (used.saturating_mul(100) / limit).min(999);
    let slider_text = format!(
        "Status: {status}\n[{filled}{empty}] {deep}/10\nCode mode: {coding_mode} (Ctrl+O)\n⚠ Higher levels use more tokens"
    );
    let deep_box = Paragraph::new(slider_text).block(
        Block::default()
            .title(format!("🧠 Deep Think | Context {used}/{limit} ({pct}%)"))
            .borders(Borders::ALL),
    );
    frame.render_widget(deep_box, footer[1]);

    if let Some(popup) = app.pending_popup_text() {
        let area = centered_rect(52, 26, frame.area());
        frame.render_widget(Clear, area);
        let popup_widget = Paragraph::new(popup).block(
            Block::default()
                .title("⚠ Confirmation Required")
                .borders(Borders::ALL),
        );
        frame.render_widget(popup_widget, area);
    }

    if let Some((sessions, selected)) = app.sessions_popup_state() {
        let area = centered_rect(70, 55, frame.area());
        frame.render_widget(Clear, area);
        let mut items: Vec<ListItem> = vec![];
        for (idx, s) in sessions.iter().enumerate() {
            let marker = if idx == selected { "▶" } else { " " };
            let style = if idx == selected {
                Style::default().fg(Color::Black).bg(Color::LightCyan)
            } else {
                Style::default().fg(Color::White)
            };
            let line = Line::from(vec![
                Span::styled("● ", Style::default().fg(Color::Red)),
                Span::styled(format!("{marker} {}", s.title), style),
            ]);
            items.push(ListItem::new(line));
        }
        if items.is_empty() {
            items.push(ListItem::new("No saved sessions found."));
        }
        let list = List::new(items).block(
            Block::default()
                .title("🗂 Sessions (↑/↓+Enter switch · click ● to delete · Esc close)")
                .borders(Borders::ALL),
        );
        frame.render_widget(list, area);
    }

    if let Some((providers, selected)) = app.provider_popup_state() {
        let area = centered_rect(50, 45, frame.area());
        frame.render_widget(Clear, area);
        let mut items: Vec<ListItem> = vec![];
        for (idx, name) in providers.iter().enumerate() {
            let marker = if idx == selected { "▶" } else { " " };
            let style = if idx == selected {
                Style::default().fg(Color::Black).bg(Color::LightCyan)
            } else {
                Style::default().fg(Color::White)
            };
            items.push(ListItem::new(Line::styled(format!("{marker} {name}"), style)));
        }
        if items.is_empty() {
            items.push(ListItem::new("No providers configured."));
        }
        let list = List::new(items).block(
            Block::default()
                .title("🔀 Switch provider (↑/↓+Enter · click to pick · Esc close)")
                .borders(Borders::ALL),
        );
        frame.render_widget(list, area);
    }

}

fn draw_alerts(frame: &mut Frame, app: &App) {
    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(CHAT_FOOTER_HEIGHT)])
        .split(frame.area());

    let sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(7), Constraint::Min(1)])
        .split(root[0]);

    let live_lines: Vec<Line> = app
        .live_tasks()
        .into_iter()
        .map(|task| Line::from(format!("▣ {task}")))
        .collect();
    let live = Paragraph::new(live_lines)
        .block(
            Block::default()
                .title("🧩 Live Tasks (running now)")
                .borders(Borders::ALL),
        )
        .wrap(Wrap { trim: true });
    frame.render_widget(live, sections[0]);

    let alerts = app.alerts();
    let mut alert_items: Vec<ListItem> = Vec::new();
    for item in alerts {
        let (icon, color) = match item.kind {
            AlertKind::Info => ("ℹ", Color::LightBlue),
            AlertKind::Success => ("✅", Color::Green),
            AlertKind::Warning => ("⚠", Color::Yellow),
            AlertKind::Error => ("⛔", Color::Red),
        };
        alert_items.push(ListItem::new(vec![
            Line::from(vec![
                Span::styled(
                    format!("{icon} {}", item.title),
                    Style::default().fg(color).add_modifier(Modifier::BOLD),
                ),
                Span::raw(format!("   [ts:{}]", item.at)),
            ]),
            Line::from(format!("   {}", item.message)),
            Line::from(""),
        ]));
    }
    if alert_items.is_empty() {
        alert_items.push(ListItem::new("No notifications yet."));
    }

    let alerts_list = List::new(alert_items).block(
        Block::default()
            .title("📣 Notifications (rectangles)")
            .borders(Borders::ALL),
    );
    frame.render_widget(alerts_list, sections[1]);

    let footer_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(65), Constraint::Percentage(35)])
        .split(root[1]);

    let left = Paragraph::new(
        "Use /alerts any time. Esc returns chat. Ctrl+C inside alerts clears notifications.",
    )
    .block(Block::default().title("Hints").borders(Borders::ALL))
    .wrap(Wrap { trim: true });
    frame.render_widget(left, footer_chunks[0]);

    let approval = if let Some((cmd, yes_selected)) = app.pending_command_for_alerts() {
        let yes_style = if yes_selected {
            Style::default().fg(Color::Black).bg(Color::Green)
        } else {
            Style::default().fg(Color::Green)
        };
        let no_style = if yes_selected {
            Style::default().fg(Color::Red)
        } else {
            Style::default().fg(Color::Black).bg(Color::Red)
        };
        Paragraph::new(Text::from(vec![
            Line::from(vec![
                Span::raw("Command request: "),
                Span::styled(cmd, Style::default().fg(Color::Yellow)),
            ]),
            Line::from(vec![
                Span::raw("Click "),
                Span::styled(" Yes ", yes_style),
                Span::raw(" or "),
                Span::styled(" No ", no_style),
                Span::raw(" (Left/Right + Enter also works)"),
            ]),
        ]))
        .block(Block::default().title("Approval").borders(Borders::ALL))
    } else {
        Paragraph::new("No command approval pending.")
            .block(Block::default().title("Approval").borders(Borders::ALL))
    };
    frame.render_widget(approval, footer_chunks[1]);
}

fn draw_settings(frame: &mut Frame, app: &App) {
    let Some(form) = app.settings_form.as_ref() else {
        return;
    };
    let area = frame.area();
    frame.render_widget(Clear, area);
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(8), Constraint::Length(8)])
        .split(area);

    let providers = Provider::all();
    let provider_name = providers[form.provider_idx].as_str();
    let model_name = providers[form.provider_idx]
        .models()
        .get(form.model_idx)
        .copied()
        .unwrap_or("unknown-model");
    let rows = [
        format!("Provider: {provider_name} (left/right)"),
        format!("Model: {model_name} (left/right)"),
        format!("Avatar: {}", form.avatar),
        format!("Cuss filter: {} (enter toggles)", if form.cuss_filter { "ON" } else { "OFF" }),
        format!("OpenAI key: {}", mask_key(&form.openai)),
        format!("Groq key: {}", mask_key(&form.groq)),
        format!("Anthropic key: {}", mask_key(&form.anthropic)),
        format!("Moonshot AI key: {}", mask_key(&form.moonshot)),
        format!("GLM (BigModel) key: {}", mask_key(&form.glm)),
        format!("Copilot token: {}", mask_key(&form.copilot)),
        format!(
            "Migrate history from: {} (←/→ · Enter to run)",
            crate::core::migration::MigrationSource::all()[form.migration_idx].label()
        ),
    ];

    let items: Vec<ListItem> = rows
        .iter()
        .enumerate()
        .map(|(idx, row)| {
            let style = if idx == form.selected {
                Style::default().fg(Color::Black).bg(Color::Yellow).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };
            ListItem::new(Line::styled(row.clone(), style))
        })
        .collect();

    let list = List::new(items).block(
        Block::default()
            .title("⚙️ Settings Tab | Ctrl+S save | F5 validate | Esc close")
            .borders(Borders::ALL),
    );
    frame.render_widget(list, chunks[0]);

    let report_text = if form.report.is_empty() {
        "No validation yet".to_string()
    } else {
        form.report.join("\n")
    };
    let report = Paragraph::new(report_text).block(
        Block::default()
            .title(format!("Validation | {}", app.last_status))
            .borders(Borders::ALL),
    );
    frame.render_widget(report, chunks[1]);
}

fn mask_key(v: &str) -> String {
    if v.is_empty() {
        "(empty)".to_string()
    } else if v.len() <= 8 {
        "********".to_string()
    } else {
        format!("{}***{}", &v[..4], &v[v.len() - 3..])
    }
}

pub fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

fn voice_streaming_line(level: u8) -> Line<'static> {
    let width = 20usize;
    let filled = ((level as usize) * width) / 100;
    let mut spans = vec![Span::styled(
        "🎙 streaming ",
        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
    )];
    spans.push(Span::raw("["));
    for i in 0..width {
        if i < filled {
            let color = if i < 7 {
                Color::Green
            } else if i < 14 {
                Color::Yellow
            } else {
                Color::Red
            };
            spans.push(Span::styled(
                "|",
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            ));
        } else {
            spans.push(Span::styled(".", Style::default().fg(Color::DarkGray)));
        }
    }
    spans.push(Span::raw("] "));
    spans.push(Span::styled(
        format!("{level:>3}%"),
        Style::default().fg(Color::LightCyan).add_modifier(Modifier::BOLD),
    ));
    Line::from(spans)
}

