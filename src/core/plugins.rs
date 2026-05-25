//! Sync plugins with Claude Code's local config.
//!
//! Claude Code tracks installed plugins in `~/.claude/plugins/installed_plugins.json`
//! (keyed by `name@marketplace`) and enabled state in `~/.claude/settings.json`
//! (`enabledPlugins`). The installable catalog lives in
//! `~/.claude/plugins/marketplaces/<marketplace>/.claude-plugin/marketplace.json`.
//!
//! `/plugins <name>` installs + enables (validated against the local catalog) and
//! `/plugout <name>` removes; both edit Claude Code's files in place, preserving
//! every other key.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};

fn claude_base() -> Result<PathBuf> {
    Ok(dirs::home_dir()
        .context("Could not resolve home directory")?
        .join(".claude"))
}

fn installed_plugins_path() -> Result<PathBuf> {
    Ok(claude_base()?
        .join("plugins")
        .join("installed_plugins.json"))
}

fn settings_path() -> Result<PathBuf> {
    Ok(claude_base()?.join("settings.json"))
}

fn marketplaces_dir() -> Result<PathBuf> {
    Ok(claude_base()?.join("plugins").join("marketplaces"))
}

/// Plugin ids currently installed in Claude Code, e.g. "frontend-design@claude-plugins-official".
pub fn read_installed() -> Vec<String> {
    let Ok(path) = installed_plugins_path() else {
        return vec![];
    };
    let Ok(text) = fs::read_to_string(&path) else {
        return vec![];
    };
    let Ok(v) = serde_json::from_str::<Value>(&text) else {
        return vec![];
    };
    let mut ids: Vec<String> = v["plugins"]
        .as_object()
        .map(|m| m.keys().cloned().collect())
        .unwrap_or_default();
    ids.sort();
    ids
}

/// Every installable plugin id from the local marketplace catalogs.
pub fn available() -> Vec<String> {
    let mut out = Vec::new();
    let Ok(dir) = marketplaces_dir() else {
        return out;
    };
    let Ok(entries) = fs::read_dir(&dir) else {
        return out;
    };
    for e in entries.filter_map(|x| x.ok()) {
        if !e.path().is_dir() {
            continue;
        }
        let Some(mp) = e.file_name().to_str().map(str::to_string) else {
            continue;
        };
        let manifest = e.path().join(".claude-plugin").join("marketplace.json");
        let Ok(text) = fs::read_to_string(&manifest) else {
            continue;
        };
        let Ok(v) = serde_json::from_str::<Value>(&text) else {
            continue;
        };
        if let Some(list) = v["plugins"].as_array() {
            for p in list {
                if let Some(name) = p["name"].as_str() {
                    out.push(format!("{name}@{mp}"));
                }
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Resolve user input (full `name@marketplace` or a bare `name`) to a catalog id.
pub fn resolve(input: &str) -> Option<String> {
    let input = input.trim();
    let avail = available();
    if avail.iter().any(|p| p == input) {
        return Some(input.to_string());
    }
    avail
        .iter()
        .find(|p| p.split('@').next() == Some(input))
        .cloned()
}

/// Install + enable a plugin in Claude Code. Returns the resolved id.
pub fn install(input: &str) -> Result<String> {
    let id = resolve(input)
        .ok_or_else(|| anyhow!("'{input}' is not in your Claude Code marketplace catalog"))?;
    let (name, marketplace) = split_id(&id)?;

    let ip = installed_plugins_path()?;
    let mut root = read_json_or(&ip, json!({"version": 2, "plugins": {}}))?;
    if !root["plugins"].is_object() {
        root["plugins"] = json!({});
    }
    let install_path = marketplaces_dir()?
        .join(&marketplace)
        .join("plugins")
        .join(&name);
    let install_path_str = install_path.to_string_lossy().to_string();
    let now = now_iso();
    root["plugins"][id.as_str()] = json!([{
        "scope": "user",
        "installPath": install_path_str,
        "version": "unknown",
        "installedAt": now.clone(),
        "lastUpdated": now,
    }]);
    write_json(&ip, &root)?;

    set_enabled(&id, true)?;
    Ok(id)
}

/// Remove + disable a plugin in Claude Code. Returns the resolved id (or raw input).
pub fn uninstall(input: &str) -> Result<String> {
    let id = resolve(input).unwrap_or_else(|| input.trim().to_string());

    let ip = installed_plugins_path()?;
    if let Ok(mut root) = read_json_or(&ip, json!({"version": 2, "plugins": {}})) {
        if let Some(obj) = root["plugins"].as_object_mut() {
            obj.remove(&id);
        }
        write_json(&ip, &root)?;
    }
    remove_enabled(&id)?;
    Ok(id)
}

fn set_enabled(id: &str, enabled: bool) -> Result<()> {
    let sp = settings_path()?;
    let mut root = read_json_or(&sp, json!({}))?;
    if !root["enabledPlugins"].is_object() {
        root["enabledPlugins"] = json!({});
    }
    root["enabledPlugins"][id] = json!(enabled);
    write_json(&sp, &root)
}

fn remove_enabled(id: &str) -> Result<()> {
    let sp = settings_path()?;
    let mut root = read_json_or(&sp, json!({}))?;
    if let Some(obj) = root["enabledPlugins"].as_object_mut() {
        obj.remove(id);
    }
    write_json(&sp, &root)
}

fn split_id(id: &str) -> Result<(String, String)> {
    let (name, mp) = id
        .split_once('@')
        .ok_or_else(|| anyhow!("Invalid plugin id (expected name@marketplace): {id}"))?;
    Ok((name.to_string(), mp.to_string()))
}

fn read_json_or(path: &Path, default: Value) -> Result<Value> {
    match fs::read_to_string(path) {
        Ok(text) => serde_json::from_str(&text)
            .with_context(|| format!("Invalid JSON in {}", path.display())),
        Err(_) => Ok(default),
    }
}

fn write_json(path: &Path, v: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let text = serde_json::to_string_pretty(v)?;
    fs::write(path, text).with_context(|| format!("Failed to write {}", path.display()))?;
    Ok(())
}

/// RFC3339 UTC timestamp (no date-crate dependency) for the install registry.
fn now_iso() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0) as i64;
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (hh, mm, ss) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    // Howard Hinnant's civil-from-days algorithm.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}.000Z")
}
