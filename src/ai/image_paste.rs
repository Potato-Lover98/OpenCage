use std::path::PathBuf;

use anyhow::{Context, Result};

pub struct ClipboardImage {
    pub path: PathBuf,
    pub size_bytes: u64,
}

pub fn grab_clipboard_image() -> Result<ClipboardImage> {
    grab_clipboard_image_impl()
}

#[cfg(feature = "native_clipboard")]
fn grab_clipboard_image_impl() -> Result<ClipboardImage> {
    let mut cb = arboard::Clipboard::new().context("Failed to access system clipboard")?;
    let img = cb
        .get_image()
        .context("No image found in clipboard (copy an image first)")?;
    let width = img.width as u32;
    let height = img.height as u32;
    let rgba = img.bytes.into_owned();
    let buf: image::RgbaImage =
        image::ImageBuffer::from_raw(width, height, rgba).context("Invalid clipboard image")?;
    let dir = tempfile::tempdir_in(std::env::temp_dir())?.keep();
    let path = dir.join("opencage_clipboard.png");
    buf.save(&path)
        .context("Failed to save clipboard image to disk")?;
    let size_bytes = std::fs::metadata(&path)
        .context("Failed to stat clipboard image file")?
        .len();
    Ok(ClipboardImage {
        path,
        size_bytes,
    })
}

#[cfg(not(feature = "native_clipboard"))]
fn grab_clipboard_image_impl() -> Result<ClipboardImage> {
    anyhow::bail!(
        "Clipboard image paste is disabled in this build (compiled without `native_clipboard`). \
         Rebuild with default features on macOS, or cross-compile with a macOS SDK and \
         `--features native_clipboard`."
    )
}

/// Copy text to the system clipboard. Prefers CLI tools (`wl-copy`/`xclip`/`xsel`) because they
/// fork and hold the selection so the paste actually works — arboard on X11 loses ownership the
/// moment its `Clipboard` is dropped. Falls back to arboard if no CLI tool is present.
pub fn copy_to_clipboard(text: &str) -> Result<()> {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let tools: [(&str, &[&str]); 3] = [
        ("wl-copy", &[]),
        ("xclip", &["-selection", "clipboard"]),
        ("xsel", &["--clipboard", "--input"]),
    ];
    for (cmd, args) in tools {
        let spawned = Command::new(cmd)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();
        if let Ok(mut child) = spawned {
            if let Some(mut stdin) = child.stdin.take() {
                if stdin.write_all(text.as_bytes()).is_ok() {
                    drop(stdin);
                    let _ = child.wait();
                    return Ok(());
                }
            }
        }
    }
    copy_to_clipboard_arboard(text)
}

#[cfg(feature = "native_clipboard")]
fn copy_to_clipboard_arboard(text: &str) -> Result<()> {
    let mut cb = arboard::Clipboard::new().context("Failed to access system clipboard")?;
    cb.set_text(text.to_string())
        .context("Failed to copy text to clipboard")?;
    Ok(())
}

#[cfg(not(feature = "native_clipboard"))]
fn copy_to_clipboard_arboard(_text: &str) -> Result<()> {
    anyhow::bail!("No clipboard tool found (install wl-clipboard, xclip, or xsel).")
}

pub fn encode_as_data_url(path: &std::path::Path) -> Result<String> {
    use base64::Engine;
    let bytes = std::fs::read(path).context("Failed to read attached image")?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    Ok(format!("data:image/png;base64,{b64}"))
}
