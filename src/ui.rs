use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap};

use crate::app::{ActiveTab, App};
use crate::core::models::Provider;

pub fn draw(frame: &mut Frame, app: &App) {
    if app.active_tab == ActiveTab::Settings {
        draw_settings(frame, app);
        return;
    }
    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(1),
            Constraint::Length(if app.show_palette { 6 } else { 0 }),
            Constraint::Length(4),
        ])
        .split(frame.area());
    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(35), Constraint::Percentage(65)])
        .split(root[0]);

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
            let marker = if idx == app.selected_file_idx { "> " } else { "  " };
            let style = if idx == app.selected_file_idx {
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
            .title(format!(
                "🌳 {} [Tab/Click]",
                app.current_dir
                    .file_name()
                    .and_then(|x| x.to_str())
                    .unwrap_or("/")
            ))
            .borders(Borders::ALL),
    );
    frame.render_widget(file_list, body[0]);

    let mut all_lines: Vec<Line> = vec![];
    for m in &app.messages {
        let (emoji, style) = match m.role.as_str() {
            "user" => ("🧑", Style::default().fg(Color::LightGreen)),
            "assistant" => ("🤖", Style::default().fg(Color::LightBlue)),
            "system" => ("🛟", Style::default().fg(Color::Magenta)),
            _ => ("💬", Style::default().fg(Color::White)),
        };
        all_lines.push(Line::from(vec![
            Span::styled(format!("{emoji} "), style.add_modifier(Modifier::BOLD)),
            Span::raw(m.content.clone()),
        ]));
    }
    let chat_height = body[1].height.saturating_sub(2) as usize;
    let total = all_lines.len();
    let scroll_from_bottom = app.chat_scroll;
    let end = total.saturating_sub(scroll_from_bottom);
    let start = end.saturating_sub(chat_height);
    let lines = if start < end && end <= total {
        all_lines[start..end].to_vec()
    } else {
        all_lines
    };
    let chat = Paragraph::new(lines)
        .block(
            Block::default()
                .title(format!(
                    "💫 {} [{}:{}]",
                    app.settings.ai_avatar,
                    app.settings.provider.as_str(),
                    app.settings.model
                ))
                .borders(Borders::ALL),
        )
        .wrap(Wrap { trim: true });
    frame.render_widget(chat, body[1]);

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
        frame.render_stateful_widget(list, root[1], &mut state);
    }

    let footer = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(root[2]);

    let input = Paragraph::new(app.input.clone()).block(
        Block::default()
            .title(format!("⌨️ Input | {}", app.last_status))
            .borders(Borders::ALL),
    );
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
        format!("Moonshot key: {}", mask_key(&form.moonshot)),
        format!("Copilot token: {}", mask_key(&form.copilot)),
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

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
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

