//! Migrate logins and chat history from other AI CLIs into OpenCage.
//!
//! Currently only Claude Code is implemented: it reads the local OAuth token
//! from `~/.claude/.credentials.json` and imports transcripts from
//! `~/.claude/projects/**/*.jsonl`. OpenClaw and Hermes are stubbed until their
//! on-disk layout is known. Nothing here touches the existing API-key settings.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use anyhow::{Context, Result, anyhow, bail};
use serde_json::Value;
use walkdir::WalkDir;

use crate::core::models::Message;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MigrationSource {
    ClaudeCode,
    OpenClaw,
    Hermes,
}

impl MigrationSource {
    pub fn all() -> [MigrationSource; 3] {
        [
            MigrationSource::ClaudeCode,
            MigrationSource::OpenClaw,
            MigrationSource::Hermes,
        ]
    }

    pub fn label(self) -> &'static str {
        match self {
            MigrationSource::ClaudeCode => "Anthropic (Claude Code)",
            MigrationSource::OpenClaw => "OpenClaw",
            MigrationSource::Hermes => "Hermes",
        }
    }

    /// Prefix for imported OpenCage session ids, e.g. `claude-<uuid>`.
    pub fn id_prefix(self) -> &'static str {
        match self {
            MigrationSource::ClaudeCode => "claude",
            MigrationSource::OpenClaw => "openclaw",
            MigrationSource::Hermes => "hermes",
        }
    }

    /// Resolve a `/migration <name>` argument to a source.
    pub fn from_arg(arg: &str) -> Option<MigrationSource> {
        match arg.trim().to_lowercase().as_str() {
            "claude" | "claude-code" | "anthropic" => Some(MigrationSource::ClaudeCode),
            "openclaw" => Some(MigrationSource::OpenClaw),
            "hermes" => Some(MigrationSource::Hermes),
            _ => None,
        }
    }
}

/// OAuth credentials lifted from a source tool's local login.
pub struct OAuthCreds {
    pub access_token: String,
    pub expires_at: Option<u64>,
}

/// One conversation imported from a source tool.
pub struct ImportedSession {
    /// Stable id from the source, so re-imports overwrite rather than duplicate.
    pub source_id: String,
    pub messages: Vec<Message>,
    pub updated_ts: u64,
}

/// Everything a single migration run produced.
pub struct MigrationOutcome {
    pub oauth: Option<OAuthCreds>,
    pub sessions: Vec<ImportedSession>,
}

pub fn migrate(source: MigrationSource) -> Result<MigrationOutcome> {
    match source {
        MigrationSource::ClaudeCode => migrate_claude_code(),
        MigrationSource::OpenClaw => bail!(
            "OpenClaw migration isn't supported yet — its credential/history layout is unknown. \
             Tell me where OpenClaw stores its login and transcripts to enable this."
        ),
        MigrationSource::Hermes => bail!(
            "Hermes migration isn't supported yet — its credential/history layout is unknown. \
             Tell me where Hermes stores its login and transcripts to enable this."
        ),
    }
}

fn claude_dir() -> Result<PathBuf> {
    Ok(dirs::home_dir()
        .context("Could not resolve home directory")?
        .join(".claude"))
}

fn migrate_claude_code() -> Result<MigrationOutcome> {
    let base = claude_dir()?;
    let oauth = read_claude_oauth(&base.join(".credentials.json")).ok();
    let sessions = read_claude_history(&base.join("projects"));
    if oauth.is_none() && sessions.is_empty() {
        bail!(
            "No Claude Code data found under {}. Is Claude Code installed and logged in here?",
            base.display()
        );
    }
    Ok(MigrationOutcome { oauth, sessions })
}

/// All Claude Code conversations currently on disk — used for the background 24/7 session sync.
pub fn claude_sessions() -> Vec<ImportedSession> {
    match claude_dir() {
        Ok(base) => read_claude_history(&base.join("projects")),
        Err(_) => Vec::new(),
    }
}

fn read_claude_oauth(path: &Path) -> Result<OAuthCreds> {
    let text =
        fs::read_to_string(path).with_context(|| format!("Failed to read {}", path.display()))?;
    let v: Value =
        serde_json::from_str(&text).with_context(|| format!("Invalid JSON in {}", path.display()))?;
    let oauth = &v["claudeAiOauth"];
    let access_token = oauth["accessToken"]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow!("No accessToken in {}", path.display()))?
        .to_string();
    // Claude Code stores expiresAt in Unix milliseconds; normalize to seconds.
    let expires_at = oauth["expiresAt"].as_u64().map(|ms| ms / 1000);
    Ok(OAuthCreds {
        access_token,
        expires_at,
    })
}

fn read_claude_history(projects_dir: &Path) -> Vec<ImportedSession> {
    let mut out = Vec::new();
    if !projects_dir.is_dir() {
        return out;
    }
    for entry in WalkDir::new(projects_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let path = entry.path();
        if path.extension().and_then(|x| x.to_str()) != Some("jsonl") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|x| x.to_str()) else {
            continue;
        };
        let Ok(text) = fs::read_to_string(path) else {
            continue;
        };
        let messages = parse_jsonl_transcript(&text);
        if messages.is_empty() {
            continue;
        }
        let updated_ts = fs::metadata(path)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        out.push(ImportedSession {
            source_id: stem.to_string(),
            messages,
            updated_ts,
        });
    }
    out
}

/// Pull plain user/assistant text out of a Claude Code `.jsonl` transcript,
/// dropping tool calls, thinking blocks, and non-message events.
fn parse_jsonl_transcript(text: &str) -> Vec<Message> {
    let mut messages = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(v) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let kind = v["type"].as_str().unwrap_or("");
        if kind != "user" && kind != "assistant" {
            continue;
        }
        let msg = &v["message"];
        let role = msg["role"].as_str().unwrap_or(kind).to_string();
        let content = extract_text(&msg["content"]);
        if content.trim().is_empty() {
            continue;
        }
        messages.push(Message { role, content });
    }
    messages
}

/// Content is either a plain string or an array of typed blocks; keep only text.
fn extract_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter(|b| b["type"].as_str() == Some("text"))
            .filter_map(|b| b["text"].as_str())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}
