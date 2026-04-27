use std::process::Command;

use anyhow::{Context, Result};

use crate::core::models::Settings;

pub fn run_in_sandbox(command_line: &str, settings: &Settings, approved: bool) -> Result<String> {
    if !approved {
        return Ok("Sandbox blocked: permission required. Use /blacklist approve on.".to_string());
    }
    let tokens: Vec<String> = command_line
        .split_whitespace()
        .map(|t| t.trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != '_'))
        .map(|s| s.to_string())
        .collect();
    if let Some(hit) = tokens
        .iter()
        .find(|t| settings.blocked_commands.contains(t.as_str()))
    {
        return Ok(format!("Sandbox blocked: '{hit}' is blacklisted"));
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
