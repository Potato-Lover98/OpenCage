//! Beetle Design: a local "live server" web UI where you describe an app and OpenCage designs it
//! as a single interactive HTML document you can preview right in the browser. "Push" saves the
//! code into the project and into RAG (the active session's memory); "Download" grabs the file.
//! Generated designs are cached under `~/.opencage/beetle/` and that cache is deleted when the
//! server stops, so nothing lingers. The page heartbeats the server, so closing the tab stops it.
//! Opened only via `/beetle`; minimal hand-rolled HTTP over `std::net` (localhost, single user).

use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::ai::providers::{SubAgent, query_with_subagent};
use crate::ai::rag::RagStore;
use crate::core::config::SettingsStore;
use crate::core::models::Settings;

const PAGE: &str = include_str!("beetle.html");
/// Stop the server if the page hasn't pinged within this window (tab closed without a beacon).
const IDLE_LIMIT: Duration = Duration::from_secs(8);

/// Temp cache for generated designs; wiped when the server stops.
fn cache_dir() -> PathBuf {
    let base = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    base.join(".opencage").join("beetle")
}

/// Serve the design UI until `stop` is set (by the app), the tab requests shutdown, or the page
/// stops heartbeating. Sets `stop` on the way out so the app can tell the server is gone, and
/// deletes the temp cache.
pub fn serve(
    listener: TcpListener,
    cwd: PathBuf,
    rag: RagStore,
    session_id: String,
    stop: Arc<AtomicBool>,
) {
    let cache = cache_dir();
    let _ = fs::create_dir_all(&cache);
    let _ = listener.set_nonblocking(true);
    let mut last_seen: Option<Instant> = None;
    while !stop.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((stream, _)) => {
                let _ = stream.set_nonblocking(false);
                let _ = stream.set_read_timeout(Some(Duration::from_secs(60)));
                let shutdown = handle(stream, &cwd, &rag, &session_id, &cache);
                // Count completion as activity so a long generation isn't mistaken for an idle tab.
                last_seen = Some(Instant::now());
                if shutdown {
                    break; // tab requested shutdown
                }
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                // No heartbeat for a while after the page first connected → tab is gone.
                if last_seen.is_some_and(|t| t.elapsed() > IDLE_LIMIT) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(120));
            }
            Err(_) => std::thread::sleep(Duration::from_millis(120)),
        }
    }
    stop.store(true, Ordering::SeqCst); // let the app observe that we've stopped
    // Storage optimization: the generated code only lives while the server runs.
    let _ = fs::remove_dir_all(&cache);
}

/// Returns `true` if the request asked the server to shut down.
fn handle(mut stream: TcpStream, cwd: &Path, rag: &RagStore, session_id: &str, cache: &Path) -> bool {
    let Some((method, path, body)) = read_request(&mut stream) else {
        return false;
    };
    match (method.as_str(), path.as_str()) {
        ("GET", "/") | ("GET", "/index.html") => {
            respond(&mut stream, "200 OK", "text/html; charset=utf-8", PAGE.as_bytes());
        }
        (_, "/api/ping") => {
            respond_json(&mut stream, &json!({ "ok": true }));
        }
        (_, "/api/shutdown") => {
            respond_json(&mut stream, &json!({ "ok": true }));
            return true;
        }
        ("POST", "/api/design") => {
            let idea = json_field(&body, "prompt").unwrap_or_default();
            let settings = SettingsStore::load_or_create().unwrap_or_default();
            let html = generate_html(&settings, &idea);
            let _ = fs::write(cache.join("last.html"), &html);
            respond_json(&mut stream, &json!({ "html": html }));
        }
        ("POST", "/api/push") => {
            let html = json_field(&body, "html").unwrap_or_default();
            let result = push_design(cwd, rag, session_id, &html);
            respond_json(&mut stream, &json!({ "result": result }));
        }
        _ => respond(&mut stream, "404 Not Found", "text/plain; charset=utf-8", b"Not found"),
    }
    false
}

/// Ask the model for a single self-contained HTML document, insisting on HTML/CSS/JS only and
/// retrying once if it returns something that isn't HTML (e.g. Python).
fn generate_html(settings: &Settings, idea: &str) -> String {
    let base = format!(
        "You are an expert front-end web developer. Build the app described below as ONE static \
         HTML file using ONLY HTML, CSS, and vanilla JavaScript — everything inline in a single \
         document. Do NOT use Python, React, frameworks, servers, build tools, or any external/CDN \
         resources. Your ENTIRE response must be a complete HTML document beginning with \
         `<!doctype html>` and ending with `</html>` — no markdown code fences, no prose, no \
         explanations, no other programming language.\n\nApp to build: {idea}"
    );
    for attempt in 0..2 {
        let prompt = if attempt == 0 {
            base.clone()
        } else {
            format!(
                "{base}\n\nYour previous answer was not valid HTML. Output ONLY the HTML document, \
                 starting with <!doctype html>. HTML/CSS/JS only — never Python."
            )
        };
        let raw = query_with_subagent(
            settings, &[], &prompt, false, false, false, 2, true, SubAgent::Coding, &[], None,
        )
        .unwrap_or_default();
        let html = extract_html(&raw);
        if looks_like_html(&html) {
            return html;
        }
    }
    "<!doctype html><html><head><meta charset=\"utf-8\"></head>\
     <body style=\"font-family:system-ui;padding:28px;color:#333\">\
     <h3>Couldn't produce HTML</h3>\
     <p>The model returned something that wasn't an HTML page. Try rephrasing, or switch to a \
     stronger model in OpenCage, then design again.</p></body></html>"
        .to_string()
}

fn looks_like_html(s: &str) -> bool {
    let l = s.to_lowercase();
    l.contains("<!doctype html") || l.contains("<html") || l.contains("<body") || l.contains("<canvas")
}

/// Write the design into the project and remember it in RAG (active session scope).
fn push_design(cwd: &Path, rag: &RagStore, session_id: &str, html: &str) -> String {
    if html.trim().is_empty() {
        return "Nothing to push — design something first.".to_string();
    }
    let dir = cwd.join("beetle");
    if let Err(e) = fs::create_dir_all(&dir) {
        return format!("Push failed: {e}");
    }
    let file = dir.join("index.html");
    if let Err(e) = fs::write(&file, html) {
        return format!("Push failed: {e}");
    }
    let scope = format!("session:{session_id}:global");
    let _ = rag.remember_scoped(&scope, "design", html);
    format!(
        "Pushed ✓ Wrote {} and saved the design to OpenCage memory (RAG).",
        file.display()
    )
}

/// Pull a clean HTML document out of the model output (handles ```html fences and prose).
fn extract_html(raw: &str) -> String {
    if let Some(fence) = raw.find("```html").map(|i| i + 7).or_else(|| raw.find("```").map(|i| i + 3))
    {
        if let Some(end) = raw[fence..].find("```") {
            return raw[fence..fence + end].trim().to_string();
        }
    }
    let lower = raw.to_lowercase();
    if let Some(start) = lower.find("<!doctype").or_else(|| lower.find("<html")) {
        if let Some(end) = lower.rfind("</html>") {
            return raw[start..end + "</html>".len()].to_string();
        }
        return raw[start..].to_string();
    }
    raw.trim().to_string()
}

fn read_request(stream: &mut TcpStream) -> Option<(String, String, String)> {
    let mut buf: Vec<u8> = Vec::new();
    let mut tmp = [0u8; 4096];
    let header_end = loop {
        let n = stream.read(&mut tmp).ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
            break pos + 4;
        }
        if buf.len() > 8_000_000 {
            return None;
        }
    };
    let header_text = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let mut header_lines = header_text.lines();
    let first = header_lines.next()?;
    let mut parts = first.split_whitespace();
    let method = parts.next()?.to_string();
    let path = parts.next()?.to_string();

    let mut content_length = 0usize;
    for line in header_lines {
        let lower = line.to_ascii_lowercase();
        if let Some(rest) = lower.strip_prefix("content-length:") {
            content_length = rest.trim().parse().unwrap_or(0);
        }
    }

    let mut body = buf[header_end..].to_vec();
    while body.len() < content_length {
        match stream.read(&mut tmp) {
            Ok(0) | Err(_) => break,
            Ok(n) => body.extend_from_slice(&tmp[..n]),
        }
    }
    body.truncate(content_length.min(body.len()));
    Some((method, path, String::from_utf8_lossy(&body).to_string()))
}

fn json_field(body: &str, key: &str) -> Option<String> {
    let v: Value = serde_json::from_str(body).ok()?;
    v.get(key)?.as_str().map(|s| s.to_string())
}

fn respond_json(stream: &mut TcpStream, v: &Value) {
    respond(stream, "200 OK", "application/json", v.to_string().as_bytes());
}

fn respond(stream: &mut TcpStream, status: &str, content_type: &str, body: &[u8]) {
    let header = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(header.as_bytes());
    let _ = stream.write_all(body);
    let _ = stream.flush();
}

fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}
