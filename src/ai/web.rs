//! Lightweight, best-effort web search (no API key) for `/web`. Scrapes DuckDuckGo's HTML
//! endpoint and returns a few result snippets to feed the model as extra context. Fragile by
//! nature (HTML can change / be rate-limited) — on any failure it just returns `None` and the
//! model answers without web context.

use std::time::Duration;

/// Search the web and return up to `max_results` formatted result snippets.
pub fn search(query: &str, max_results: usize) -> Option<String> {
    let query = query.trim();
    if query.is_empty() {
        return None;
    }
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(12))
        .user_agent("Mozilla/5.0 (X11; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0")
        .build()
        .ok()?;
    let html = client
        .post("https://html.duckduckgo.com/html/")
        .form(&[("q", query)])
        .send()
        .ok()?
        .text()
        .ok()?;

    let mut results = Vec::new();
    for chunk in html.split("result__a").skip(1) {
        let title = chunk
            .split_once('>')
            .and_then(|(_, rest)| rest.split_once('<'))
            .map(|(t, _)| strip_html(t))
            .unwrap_or_default();
        if title.trim().is_empty() {
            continue;
        }
        let snippet = chunk
            .split("result__snippet")
            .nth(1)
            .and_then(|s| s.split_once('>'))
            .and_then(|(_, rest)| rest.split_once("</a"))
            .map(|(t, _)| strip_html(t))
            .unwrap_or_default();
        if snippet.trim().is_empty() {
            results.push(format!("- {title}"));
        } else {
            results.push(format!("- {title}: {snippet}"));
        }
        if results.len() >= max_results {
            break;
        }
    }

    if results.is_empty() {
        None
    } else {
        Some(format!(
            "Live web search results for \"{query}\" (use these for up-to-date facts):\n{}",
            results.join("\n")
        ))
    }
}

/// Strip HTML tags and decode a few common entities; collapse whitespace.
fn strip_html(s: &str) -> String {
    let mut text = String::new();
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => text.push(c),
            _ => {}
        }
    }
    text.replace("&amp;", "&")
        .replace("&#x27;", "'")
        .replace("&#x2F;", "/")
        .replace("&quot;", "\"")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}
