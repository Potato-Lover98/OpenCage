use std::process::Command;

use anyhow::{Context, Result};

use crate::core::models::Settings;

pub fn run_in_sandbox(command_line: &str, settings: &Settings, approved: bool) -> Result<String> {
    if !approved {
        return Ok("Sandbox blocked: permission required. Use /blacklist approve on.".to_string());
    }
    let first = command_line.split_whitespace().next().unwrap_or_default();
    if settings.blocked_commands.contains(first) {
        return Ok(format!("Sandbox blocked: '{first}' is blacklisted"));
    }
    let output = Command::new("bash")
        .arg("-lc")
        .arg(command_line)
        .output()
        .context("Failed to execute command")?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    Ok(format!("stdout:\n{stdout}\nstderr:\n{stderr}"))
}
