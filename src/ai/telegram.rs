//! Telegram bridge: long-poll a bot and answer each message with the configured AI provider,
//! so the user can chat with — and remotely drive — OpenCage from Telegram. A coding request
//! runs the autonomous-coding pipeline on the host computer; the folder-trust gate is surfaced
//! as inline Yes/No buttons in Telegram. Started/stopped via `/bots`; the token lives (encrypted)
//! in settings, never hard-coded.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::time::Duration;

use reqwest::blocking::Client;
use serde_json::{Value, json};

use crate::ai::providers::{SubAgent, query_with_subagent};
use crate::core::config::SettingsStore;
use crate::core::models::{Message, Settings};

const POLL_SECS: u64 = 25;
const MAX_HISTORY: usize = 20;
const TELEGRAM_MAX_CHARS: usize = 4096;

/// Long-poll Telegram until `stop` is set. Text messages are answered by the AI provider;
/// coding requests run on the host in `cwd` after a folder-trust confirmation (inline buttons).
pub fn run_bot(
    token: String,
    mut settings: Settings,
    cwd: PathBuf,
    stop: Arc<AtomicBool>,
    log: Sender<String>,
) {
    let client = Client::builder()
        .timeout(Duration::from_secs(POLL_SECS + 35))
        .build()
        .unwrap_or_else(|_| Client::new());
    let base = format!("https://api.telegram.org/bot{token}");

    match client
        .get(format!("{base}/getMe"))
        .send()
        .and_then(|r| r.json::<Value>())
    {
        Ok(v) if v["ok"].as_bool() == Some(true) => {
            let uname = v["result"]["username"].as_str().unwrap_or("your bot");
            let _ = log.send(format!(
                "Telegram bot @{uname} connected — message it to chat with OpenCage."
            ));
        }
        _ => {
            let _ = log.send(
                "Telegram: token rejected (getMe failed). Check the token and try /bots again."
                    .to_string(),
            );
            return;
        }
    }

    let mut offset: i64 = 0;
    // Skip any backlog so we only answer messages sent after the bot starts.
    if let Ok(v) = client
        .get(format!("{base}/getUpdates?offset=-1&timeout=0"))
        .send()
        .and_then(|r| r.json::<Value>())
        && let Some(last) = v["result"].as_array().and_then(|a| a.last())
        && let Some(id) = last["update_id"].as_i64()
    {
        offset = id + 1;
    }

    let mut histories: HashMap<i64, Vec<Message>> = HashMap::new();
    // Coding prompts awaiting a folder-trust button tap, keyed by chat id.
    let mut pending: HashMap<i64, String> = HashMap::new();

    while !stop.load(Ordering::Relaxed) {
        let url = format!("{base}/getUpdates?timeout={POLL_SECS}&offset={offset}");
        let resp = match client.get(&url).send().and_then(|r| r.json::<Value>()) {
            Ok(v) => v,
            Err(_) => {
                std::thread::sleep(Duration::from_secs(2));
                continue;
            }
        };
        let Some(updates) = resp["result"].as_array() else {
            continue;
        };
        if updates.is_empty() {
            continue;
        }
        // Pick up changes made in the TUI (provider/model switch, key edits, trust) so Telegram
        // always uses the same provider as the computer. The store is the shared source of truth.
        if let Ok(latest) = SettingsStore::load_or_create() {
            settings = latest;
        }
        for u in updates {
            if let Some(id) = u["update_id"].as_i64() {
                offset = id + 1;
            }
            if stop.load(Ordering::Relaxed) {
                break;
            }

            // Inline-button taps: the folder-trust confirmation for remote coding.
            if let Some(cq) = u.get("callback_query") {
                let data = cq["data"].as_str().unwrap_or("");
                let chat_id = cq["message"]["chat"]["id"].as_i64().unwrap_or_default();
                answer_callback(&client, &base, cq["id"].as_str().unwrap_or(""));
                match data {
                    "trust_yes" => {
                        if let Some(prompt) = pending.remove(&chat_id) {
                            settings.trusted_paths.insert(cwd.to_string_lossy().to_string());
                            let _ = SettingsStore::save(&settings);
                            let _ = send_message(&client, &base, chat_id, "✅ Folder trusted — coding now.");
                            run_coding(&client, &base, chat_id, &settings, &cwd, &prompt, &log);
                        } else {
                            let _ = send_message(&client, &base, chat_id, "Nothing is waiting for confirmation.");
                        }
                    }
                    "trust_no" => {
                        pending.remove(&chat_id);
                        let _ = send_message(&client, &base, chat_id, "❌ Cancelled — folder left untrusted.");
                    }
                    _ => {}
                }
                continue;
            }

            let msg = &u["message"];
            let (Some(text), Some(chat_id)) = (msg["text"].as_str(), msg["chat"]["id"].as_i64())
            else {
                continue;
            };
            let text = text.trim().to_string();
            if text.is_empty() {
                continue;
            }

            if text == "/start" || text == "/help" {
                let _ = send_message(
                    &client,
                    &base,
                    chat_id,
                    "OpenCage here 👋 Chat normally, or ask me to build something and I'll code it on \
                     the host computer — I'll ask you to confirm the folder first.",
                );
                continue;
            }
            if text.starts_with("!run") {
                let _ = send_message(&client, &base, chat_id, "Shell commands aren't allowed over Telegram.");
                continue;
            }

            let _ = log.send(format!("📥 Telegram [{chat_id}]: {text}"));

            // Coding request → run on the host (gated by folder trust).
            if crate::app::is_coding_task(&text) {
                let trusted = settings
                    .trusted_paths
                    .contains(&cwd.to_string_lossy().to_string());
                if trusted {
                    run_coding(&client, &base, chat_id, &settings, &cwd, &text, &log);
                } else {
                    pending.insert(chat_id, text);
                    let _ = send_trust_prompt(&client, &base, chat_id, &cwd);
                }
                continue;
            }

            // Plain chat.
            let history = histories.entry(chat_id).or_default();
            let reply = query_with_subagent(
                &settings, history, &text, false, false, false, 2, false, SubAgent::General, &[],
                None,
            )
            .unwrap_or_else(|e| format!("Provider error: {e}"));
            history.push(Message {
                role: "user".to_string(),
                content: text,
            });
            history.push(Message {
                role: "assistant".to_string(),
                content: reply.clone(),
            });
            if history.len() > MAX_HISTORY {
                let drop = history.len() - MAX_HISTORY;
                history.drain(0..drop);
            }
            let _ = send_message(&client, &base, chat_id, &reply);
            let _ = log.send(format!("📤 Telegram [{chat_id}]: replied"));
        }
    }
    let _ = log.send("Telegram bot stopped.".to_string());
}

/// Run the autonomous-coding pipeline on the host and report the result over Telegram.
/// Files are written under `cwd`; any shell commands are reported, not executed remotely.
fn run_coding(
    client: &Client,
    base: &str,
    chat_id: i64,
    settings: &Settings,
    cwd: &Path,
    prompt: &str,
    log: &Sender<String>,
) {
    let _ = send_message(client, base, chat_id, "🛠 Coding on your computer…");
    let _ = log.send(format!("Telegram: coding requested → {prompt}"));
    match crate::app::run_autonomous_coding(settings, &[], prompt, false, 2, false, cwd) {
        Ok(r) => {
            let mut out = r.text;
            if !r.commands.is_empty() {
                out.push_str("\n\nPlanned commands (review/run in OpenCage — not run from Telegram):");
                for c in &r.commands {
                    out.push_str(&format!("\n• {c}"));
                }
            }
            let _ = send_message(client, base, chat_id, &out);
            let _ = log.send(format!("Telegram: coding done for [{chat_id}]"));
        }
        Err(e) => {
            let _ = send_message(client, base, chat_id, &format!("Coding failed: {e}"));
        }
    }
}

fn send_trust_prompt(client: &Client, base: &str, chat_id: i64, cwd: &Path) -> bool {
    let body = json!({
        "chat_id": chat_id,
        "text": format!("Trust this folder for autonomous coding and proceed?\n{}", cwd.display()),
        "reply_markup": {
            "inline_keyboard": [[
                {"text": "✅ Yes", "callback_data": "trust_yes"},
                {"text": "❌ No", "callback_data": "trust_no"}
            ]]
        }
    });
    client
        .post(format!("{base}/sendMessage"))
        .json(&body)
        .send()
        .is_ok()
}

fn answer_callback(client: &Client, base: &str, callback_id: &str) {
    if callback_id.is_empty() {
        return;
    }
    let _ = client
        .post(format!("{base}/answerCallbackQuery"))
        .json(&json!({ "callback_query_id": callback_id }))
        .send();
}

fn send_message(client: &Client, base: &str, chat_id: i64, text: &str) -> bool {
    let body = json!({ "chat_id": chat_id, "text": clamp(text, TELEGRAM_MAX_CHARS) });
    client
        .post(format!("{base}/sendMessage"))
        .json(&body)
        .send()
        .is_ok()
}

fn clamp(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        s.chars().take(max - 1).collect::<String>() + "…"
    }
}
