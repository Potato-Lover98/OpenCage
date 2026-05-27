use std::collections::HashSet;
use std::process::Command;

use anyhow::{Context, Result};

use crate::core::models::Settings;

pub fn run_in_sandbox(command_line: &str, settings: &Settings, approved: bool) -> Result<String> {
    if !approved {
        return Ok("Sandbox blocked: permission required. Use /blacklist approve on.".to_string());
    }
    if let Some(hit) = blacklisted_command(command_line, &settings.blocked_commands) {
        return Ok(format!("I can't run that — '{hit}' is in your blacklist."));
    }

    let output = Command::new("bash")
        .arg("-lc")
        .arg(command_line)
        .output()
        .context("Failed to execute command")?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut out = String::new();
    if !stdout.trim().is_empty() {
        out.push_str(&stdout);
    }
    if !stderr.trim().is_empty() {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&stderr);
    }
    Ok(out)
}

/// Find a blacklisted command anywhere in the line. Splits on shell separators and checks both
/// the token and its basename, so `rm`, `sudo rm`, `/bin/rm`, and `ls | rm` are all caught.
pub fn blacklisted_command(command_line: &str, blocked: &HashSet<String>) -> Option<String> {
    for raw in command_line.split(|c: char| {
        c.is_whitespace() || matches!(c, ';' | '|' | '&' | '(' | ')' | '<' | '>' | '`')
    }) {
        let tok = raw.trim_matches(|c: char| {
            !c.is_alphanumeric() && c != '-' && c != '_' && c != '.' && c != '/'
        });
        if tok.is_empty() {
            continue;
        }
        if blocked.contains(tok) {
            return Some(tok.to_string());
        }
        if let Some(base) = tok.rsplit('/').next() {
            if base != tok && blocked.contains(base) {
                return Some(base.to_string());
            }
        }
    }
    None
}
