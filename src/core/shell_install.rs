use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

const BEGIN_MARKER: &str = "# >>> opencage auto-launch >>>";
const END_MARKER: &str = "# <<< opencage auto-launch <<<";

pub fn ensure_shell_launch_command() -> Result<()> {
    let exe = std::env::current_exe().context("Failed to resolve current executable path")?;
    let exe = fs::canonicalize(exe).context("Failed to canonicalize executable path")?;
    let home = dirs::home_dir().context("Failed to resolve home directory")?;

    for rc_name in [".bashrc", ".zshrc"] {
        upsert_rc_block(&home.join(rc_name), &exe)?;
    }
    Ok(())
}

fn upsert_rc_block(rc_path: &Path, exe: &Path) -> Result<()> {
    let existing = fs::read_to_string(rc_path).unwrap_or_default();
    let block = format!(
        "{BEGIN_MARKER}\n# Added by Opencage so `opencage` runs this binary.\nalias opencage=\"{}\"\n{END_MARKER}\n",
        escape_path_for_shell(exe)
    );

    let updated = if let (Some(start), Some(end)) = (existing.find(BEGIN_MARKER), existing.find(END_MARKER)) {
        let end_idx = end + END_MARKER.len();
        let mut out = String::new();
        out.push_str(&existing[..start]);
        if !out.ends_with('\n') {
            out.push('\n');
        }
        out.push_str(&block);
        out.push_str(existing[end_idx..].trim_start_matches('\n'));
        out
    } else {
        let mut out = existing;
        if !out.is_empty() && !out.ends_with('\n') {
            out.push('\n');
        }
        out.push('\n');
        out.push_str(&block);
        out
    };

    fs::write(rc_path, updated)
        .with_context(|| format!("Failed to update shell profile {}", rc_path.display()))?;
    Ok(())
}

fn escape_path_for_shell(path: &Path) -> String {
    let s = path.to_string_lossy().to_string();
    s.replace('"', "\\\"")
}
