use anyhow::{Context, Result};
use reqwest::blocking::Client;
use serde_json::json;

use crate::ai::sandbox::run_in_sandbox;
use crate::core::models::{Message, Provider, Settings};

/// How much conversation history (in chars) to send as context — a large budget so OpenCage
/// uses a huge context like Claude Code, rather than a tiny fixed message count. ~150k tokens,
/// which fits safely inside every supported model's window (all ≥ 200k tokens).
pub const CONTEXT_BUDGET_CHARS: usize = 600_000;

/// The most-recent messages whose combined size fits `budget_chars`, in chronological order.
/// The latest message is always included even if it alone exceeds the budget.
pub fn recent_within_budget(history: &[Message], budget_chars: usize) -> Vec<&Message> {
    let mut total = 0usize;
    let mut picked: Vec<&Message> = Vec::new();
    for m in history.iter().rev() {
        let cost = m.role.len() + m.content.len() + 8;
        if !picked.is_empty() && total + cost > budget_chars {
            break;
        }
        total += cost;
        picked.push(m);
    }
    picked.reverse();
    picked
}

pub fn validate_settings_keys(settings: &Settings) -> Vec<String> {
    vec![
        validate_one("OpenAI", settings.openai_api_key.as_deref(), &["sk-"]),
        validate_one("Groq", settings.groq_api_key.as_deref(), &["gsk_"]),
        validate_one("Anthropic", settings.anthropic_api_key.as_deref(), &["sk-ant-"]),
        validate_one("Moonshot", settings.moonshot_api_key.as_deref(), &["sk-", "moon-"]),
        validate_glm_key(settings.glm_api_key.as_deref()),
        validate_one(
            "GitHub Copilot",
            settings.github_copilot_token.as_deref(),
            &["ghu_", "github_pat_", "ghp_"],
        ),
    ]
}

fn validate_glm_key(key: Option<&str>) -> String {
    match key {
        None => "GLM (BigModel): missing".to_string(),
        Some(v) if v.trim().is_empty() => "GLM (BigModel): empty".to_string(),
        Some(v) => {
            let v = v.trim();
            let ok_dot = v.contains('.') && v.len() >= 40;
            let ok_long = v.len() >= 24;
            if ok_dot || ok_long {
                "GLM (BigModel): good format".to_string()
            } else {
                "GLM (BigModel): key looks short or unusual".to_string()
            }
        }
    }
}

fn validate_one(name: &str, key: Option<&str>, prefixes: &[&str]) -> String {
    match key {
        None => format!("{name}: missing"),
        Some(v) if v.trim().is_empty() => format!("{name}: empty"),
        Some(v) => {
            if prefixes.iter().any(|p| v.starts_with(p)) && v.len() >= 16 {
                format!("{name}: good format")
            } else {
                format!("{name}: bad format")
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum SubAgent {
    General,
    Coding,
    Research,
    Reviewer,
    Shell,
}

impl SubAgent {
    pub fn name(self) -> &'static str {
        match self {
            SubAgent::General => "general",
            SubAgent::Coding => "coding",
            SubAgent::Research => "research",
            SubAgent::Reviewer => "reviewer",
            SubAgent::Shell => "shell",
        }
    }

    fn system_prompt(self, avatar: &str, buddy_mode: bool) -> String {
        let base = if buddy_mode {
            format!("You are {avatar}, a concise and supportive coding buddy.")
        } else {
            format!("You are {avatar}, a capable software agent.")
        };
        let specialist = match self {
            SubAgent::General => "Handle general product and coding questions.",
            SubAgent::Coding => "Produce concrete, runnable code with clear implementation steps, sensible structure, and practical defaults.",
            SubAgent::Research => "Explain concepts, compare options, and answer side questions.",
            SubAgent::Reviewer => "Review for risks, bugs, and missing test coverage.",
            SubAgent::Shell => "Provide safe shell guidance and command plans.",
        };
        format!("{base} You are acting as the {specialist}")
    }
}

pub fn select_subagents(prompt: &str) -> Vec<SubAgent> {
    let mut picked = vec![route_subagent(prompt)];
    let p = prompt.to_lowercase();
    let word_count = p.split_whitespace().count();
    let complexity_hits = [
        "architecture",
        "multi-step",
        "end-to-end",
        "production",
        "security",
        "optimize",
        "performance",
        "migration",
        "design",
        "system",
    ]
    .iter()
    .filter(|k| p.contains(**k))
    .count();

    if (p.contains("implement") || p.contains("build") || p.contains("refactor"))
        && (p.contains("why") || p.contains("tradeoff") || p.contains("compare"))
    {
        picked.push(SubAgent::Research);
    }
    if p.contains("review") || p.contains("safe") || p.contains("risk") {
        picked.push(SubAgent::Reviewer);
    }
    if word_count > 30 || complexity_hits >= 2 {
        picked.push(SubAgent::Coding);
        picked.push(SubAgent::Research);
        picked.push(SubAgent::Reviewer);
    }
    picked.sort_by_key(|a| a.name());
    picked.dedup_by_key(|a| a.name());
    picked
}

pub fn query_with_subagent(
    settings: &Settings,
    history: &[Message],
    prompt: &str,
    buddy_mode: bool,
    command_approval: bool,
    deep_think_enabled: bool,
    deep_think_level: u8,
    coding_expanded: bool,
    subagent: SubAgent,
    rag_context: &[String],
    attached_image_data_url: Option<&str>,
) -> Result<String> {
    if let Some(rest) = prompt.strip_prefix("!run ") {
        return run_in_sandbox(rest, settings, command_approval);
    }
    let base_prompt = subagent.system_prompt(&settings.ai_avatar, buddy_mode);
    let depth_hint = if !deep_think_enabled {
        "DeepThink is disabled."
    } else if deep_think_level >= 7 {
        "DeepThink mode is high. Reason carefully, explore alternatives, and provide higher-rigor answers."
    } else if deep_think_level >= 4 {
        "DeepThink mode is medium. Provide balanced depth and concise reasoning."
    } else {
        "DeepThink mode is low. Keep responses concise."
    };
    let coding_hint = if matches!(subagent, SubAgent::Coding) {
        if coding_expanded {
            "For coding outputs, provide expanded and well-documented implementation details."
        } else {
            "For coding outputs, keep code concise and minimal while still correct."
        }
    } else {
        ""
    };
    let system_prompt = format!("{base_prompt} {depth_hint} {coding_hint}");
    let rag_text = if rag_context.is_empty() {
        String::new()
    } else {
        format!(
            "Known user memory (RAG):\n{}\n",
            rag_context
                .iter()
                .map(|v| format!("- {v}"))
                .collect::<Vec<_>>()
                .join("\n")
        )
    };

    let mut messages = vec![json!({
        "role": "system",
        "content": format!("{system_prompt}\n{rag_text}")
    })];
    // With an image attached, keep the payload lean (just the latest turn); otherwise send as
    // much recent history as fits the context budget.
    let visible_history: Vec<&Message> = if attached_image_data_url.is_some() {
        history.iter().rev().take(1).collect::<Vec<_>>().into_iter().rev().collect()
    } else {
        recent_within_budget(history, CONTEXT_BUDGET_CHARS)
    };
    for m in &visible_history {
        messages.push(json!({"role": m.role, "content": m.content}));
    }
    if let Some(url) = attached_image_data_url {
        messages.push(json!({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": url}}
            ]
        }));
    }

    let client = Client::new();
    let answer = call_with_fallback(&client, settings, history, subagent, &messages, &system_prompt);
    Ok(format!("[{}] {}", settings.ai_avatar, answer))
}

fn call_with_fallback(
    client: &Client,
    settings: &Settings,
    history: &[Message],
    subagent: SubAgent,
    messages: &[serde_json::Value],
    system_prompt: &str,
) -> String {
    let mut tried: Vec<Provider> = Vec::new();
    let first = select_provider_for_subagent(settings, subagent);
    let mut queue: Vec<ProviderAssignment> = vec![first.clone()];
    for p in enabled_with_credentials(settings) {
        if p != first.provider {
            queue.push(ProviderAssignment {
                model: normalized_model_for_provider(settings, &p),
                provider: p,
            });
        }
    }
    let mut last_err = String::from("No providers available.");
    for assignment in queue {
        if tried.contains(&assignment.provider) {
            continue;
        }
        tried.push(assignment.provider.clone());
        let result = match assignment.provider {
            Provider::OpenAi => call_openai(client, settings, &assignment.model, messages),
            Provider::Groq => call_groq(client, settings, &assignment.model, messages),
            Provider::Anthropic => {
                call_anthropic(client, settings, &assignment.model, history, system_prompt)
            }
            Provider::MoonshotAi => call_moonshot(client, settings, &assignment.model, messages),
            Provider::GlmBigModel => call_glm_bigmodel(client, settings, &assignment.model, messages),
            Provider::GithubCopilot => call_copilot(client, settings, &assignment.model, messages),
        };
        match result {
            Ok(text) if !response_is_error(&text) => return text,
            Ok(text) => {
                last_err = format!("[{}] {}", assignment.provider.as_str(), text);
            }
            Err(e) => {
                last_err = format!("[{}] {}", assignment.provider.as_str(), e);
            }
        }
    }
    format!("All providers failed. Last: {last_err}")
}

fn response_is_error(text: &str) -> bool {
    let t = text.to_lowercase();
    t.contains("api error")
        || t.contains("returned no content")
        || t.contains("checking router")
        || t.contains("no content")
        || t.contains("invalid json")
        || t.contains("no model routes")
        || t.trim().is_empty()
}

pub fn query_coding_actions(
    settings: &Settings,
    history: &[Message],
    prompt: &str,
    deep_think_enabled: bool,
    deep_think_level: u8,
    coding_expanded: bool,
) -> Result<String> {
    let depth_hint = if !deep_think_enabled {
        "Keep reasoning concise."
    } else if deep_think_level >= 7 {
        "Reason deeply and plan multi-step edits."
    } else {
        "Use balanced reasoning depth."
    };
    let detail_hint = if coding_expanded {
        "Return complete implementation edits."
    } else {
        "Prefer smaller, focused edits."
    };
    let system_prompt = format!(
        "You are an autonomous coding agent working in a local project directory. \
Output the file blocks FIRST, before any prose. Write each file EXACTLY like:
<OPENCAGE_FILE path=\"relative/path.ext\">
...complete file content...
</OPENCAGE_FILE>
For terminal commands use:
<OPENCAGE_CMD>command here</OPENCAGE_CMD>
Rules: write COMPLETE file contents (never elide with comments like \"// rest unchanged\"); \
do NOT wrap the blocks in markdown code fences; output only plain UTF-8 text; \
keep any summary to one short sentence placed AFTER all blocks. \
You may include multiple file/cmd blocks. {depth_hint} {detail_hint}"
    );
    let mut messages = vec![json!({"role":"system","content":system_prompt})];
    for m in recent_within_budget(history, CONTEXT_BUDGET_CHARS) {
        messages.push(json!({"role": m.role, "content": m.content}));
    }
    messages.push(json!({"role":"user","content":prompt}));

    let client = Client::new();
    let assignment = select_provider_for_subagent(settings, SubAgent::Coding);
    match assignment.provider {
        Provider::OpenAi => call_openai(&client, settings, &assignment.model, &messages),
        Provider::Groq => call_groq(&client, settings, &assignment.model, &messages),
        Provider::Anthropic => {
            call_anthropic(&client, settings, &assignment.model, history, &system_prompt)
        }
        Provider::MoonshotAi => call_moonshot(&client, settings, &assignment.model, &messages),
        Provider::GlmBigModel => call_glm_bigmodel(&client, settings, &assignment.model, &messages),
        Provider::GithubCopilot => call_copilot(&client, settings, &assignment.model, &messages),
    }
}

#[derive(Clone)]
struct ProviderAssignment {
    provider: Provider,
    model: String,
}

fn select_provider_for_subagent(settings: &Settings, subagent: SubAgent) -> ProviderAssignment {
    let enabled = enabled_with_credentials(settings);
    // Honor the user's explicitly chosen provider first when it has usable credentials;
    // the remaining providers stay available as fallback in call_with_fallback.
    if enabled.iter().any(|p| *p == settings.provider) {
        return ProviderAssignment {
            model: normalized_model_for_provider(settings, &settings.provider),
            provider: settings.provider.clone(),
        };
    }
    let ordered = preferred_order(subagent);
    for preferred in ordered {
        if enabled.iter().any(|p| *p == preferred) {
            return ProviderAssignment {
                model: normalized_model_for_provider(settings, &preferred),
                provider: preferred,
            };
        }
    }
    let fallback = enabled
        .first()
        .cloned()
        .unwrap_or_else(|| settings.provider.clone());
    ProviderAssignment {
        model: normalized_model_for_provider(settings, &fallback),
        provider: fallback,
    }
}

fn enabled_with_credentials(settings: &Settings) -> Vec<Provider> {
    // The selected provider is authoritative: only ever use it, so switching providers never
    // leaves a previously-selected one connected. If it lacks credentials the call surfaces a
    // clear error rather than silently falling back to a different provider (e.g. the migrated
    // Anthropic OAuth being used while Groq is selected).
    vec![settings.provider.clone()]
}

fn preferred_order(subagent: SubAgent) -> Vec<Provider> {
    match subagent {
        SubAgent::Research => vec![
            Provider::Anthropic,
            Provider::OpenAi,
            Provider::MoonshotAi,
            Provider::GlmBigModel,
            Provider::Groq,
            Provider::GithubCopilot,
        ],
        SubAgent::Reviewer => vec![
            Provider::Anthropic,
            Provider::OpenAi,
            Provider::Groq,
            Provider::MoonshotAi,
            Provider::GlmBigModel,
            Provider::GithubCopilot,
        ],
        SubAgent::Coding => vec![
            Provider::OpenAi,
            Provider::Groq,
            Provider::GithubCopilot,
            Provider::Anthropic,
            Provider::MoonshotAi,
            Provider::GlmBigModel,
        ],
        SubAgent::Shell => vec![
            Provider::OpenAi,
            Provider::Groq,
            Provider::Anthropic,
            Provider::GithubCopilot,
            Provider::MoonshotAi,
            Provider::GlmBigModel,
        ],
        SubAgent::General => vec![
            Provider::OpenAi,
            Provider::Anthropic,
            Provider::Groq,
            Provider::MoonshotAi,
            Provider::GlmBigModel,
            Provider::GithubCopilot,
        ],
    }
}

pub fn provider_has_credentials(settings: &Settings, provider: &Provider) -> bool {
    match provider {
        Provider::OpenAi => settings
            .openai_api_key
            .as_ref()
            .is_some_and(|v| !v.trim().is_empty())
            || std::env::var("OPENAI_API_KEY").ok().is_some(),
        Provider::Groq => settings
            .groq_api_key
            .as_ref()
            .is_some_and(|v| !v.trim().is_empty())
            || std::env::var("GROQ_API_KEY").ok().is_some(),
        Provider::Anthropic => settings
            .anthropic_api_key
            .as_ref()
            .is_some_and(|v| !v.trim().is_empty())
            || settings
                .anthropic_oauth_token
                .as_ref()
                .is_some_and(|v| !v.trim().is_empty())
            || std::env::var("ANTHROPIC_API_KEY").ok().is_some(),
        Provider::MoonshotAi => settings
            .moonshot_api_key
            .as_ref()
            .is_some_and(|v| !v.trim().is_empty())
            || std::env::var("MOONSHOT_API_KEY").ok().is_some(),
        Provider::GlmBigModel => settings
            .glm_api_key
            .as_ref()
            .is_some_and(|v| !v.trim().is_empty())
            || std::env::var("ZHIPU_API_KEY").ok().is_some()
            || std::env::var("GLM_API_KEY").ok().is_some(),
        Provider::GithubCopilot => settings
            .github_copilot_token
            .as_ref()
            .is_some_and(|v| !v.trim().is_empty())
            || std::env::var("GITHUB_COPILOT_TOKEN").ok().is_some(),
    }
}

fn normalized_model_for_provider(settings: &Settings, provider: &Provider) -> String {
    if let Some(m) = settings.provider_models.get(provider.as_str()) {
        if provider.models().iter().any(|v| *v == m) {
            return m.clone();
        }
    }
    if provider == &settings.provider && provider.models().iter().any(|m| *m == settings.model) {
        return settings.model.clone();
    }
    provider
        .models()
        .first()
        .copied()
        .unwrap_or("unknown-model")
        .to_string()
}

fn route_subagent(prompt: &str) -> SubAgent {
    let p = prompt.to_lowercase();
    if p.starts_with("!run ") || p.contains("terminal") || p.contains("shell") {
        SubAgent::Shell
    } else if p.contains("review") || p.contains("audit") || p.contains("bug risk") {
        SubAgent::Reviewer
    } else if p.contains("why") || p.contains("explain") || p.contains("compare") || p.contains("/btw") {
        SubAgent::Research
    } else if p.contains("implement")
        || p.contains("write code")
        || p.contains("refactor")
        || p.contains("build")
    {
        SubAgent::Coding
    } else {
        SubAgent::General
    }
}

fn call_openai(
    client: &Client,
    settings: &Settings,
    model: &str,
    messages: &[serde_json::Value],
) -> Result<String> {
    let key = settings
        .openai_api_key
        .clone()
        .or_else(|| std::env::var("OPENAI_API_KEY").ok())
        .context("OPENAI_API_KEY missing")?;
    let body = json!({"model": model, "messages": messages});
    let v: serde_json::Value = client
        .post("https://api.openai.com/v1/chat/completions")
        .bearer_auth(key)
        .json(&body)
        .send()
        .context("OpenAI request failed")?
        .json()
        .context("OpenAI returned invalid JSON")?;
    extract_openai_like_content("OpenAI", &v)
}

fn call_groq(
    client: &Client,
    settings: &Settings,
    model: &str,
    messages: &[serde_json::Value],
) -> Result<String> {
    let key = settings
        .groq_api_key
        .clone()
        .or_else(|| std::env::var("GROQ_API_KEY").ok())
        .context("GROQ_API_KEY missing")?;
    let body = json!({"model": model, "messages": messages});
    let resp = client
        .post("https://api.groq.com/openai/v1/chat/completions")
        .bearer_auth(key)
        .json(&body)
        .send()
        .context("Groq request failed")?;
    let status = resp.status();
    let v: serde_json::Value = resp
        .json()
        .context("Groq returned invalid JSON")?;
    if !status.is_success() {
        let err = v["error"]["message"]
            .as_str()
            .or_else(|| v["message"].as_str())
            .unwrap_or("Unknown Groq error");
        return Ok(format!("Groq API error ({status}): {err}"));
    }
    extract_openai_like_content("Groq", &v)
}

fn call_anthropic(
    client: &Client,
    settings: &Settings,
    model: &str,
    history: &[Message],
    system_prompt: &str,
) -> Result<String> {
    let anthropic_msgs: Vec<serde_json::Value> = history
        .iter()
        .map(|m| {
            json!({
                "role": if m.role == "assistant" { "assistant" } else { "user" },
                "content": m.content
            })
        })
        .collect();

    // Try the requested model; if Anthropic 404s it (e.g. a 4.7 tier that isn't live yet),
    // transparently retry with the latest available model of the same tier.
    let mut models = vec![model.to_string()];
    if let Some(fb) = anthropic_fallback_model(model) {
        models.push(fb.to_string());
    }
    let mut last = "Anthropic returned no usable response".to_string();
    for m in &models {
        let (status, v) = anthropic_send(client, settings, m, system_prompt, &anthropic_msgs)?;
        if status.is_success() {
            // Concatenate every text block (a response may contain more than one).
            let text: String = v["content"]
                .as_array()
                .map(|blocks| {
                    blocks
                        .iter()
                        .filter(|b| b["type"].as_str() == Some("text"))
                        .filter_map(|b| b["text"].as_str())
                        .collect::<Vec<_>>()
                        .join("")
                })
                .unwrap_or_default();
            return Ok(if text.trim().is_empty() {
                "No content".to_string()
            } else {
                text
            });
        }
        let err = v["error"]["message"]
            .as_str()
            .unwrap_or("Unknown Anthropic error");
        last = format!("Anthropic API error ({status}): {err}");
        if status.as_u16() != 404 {
            break; // only model-not-found is worth a fallback attempt
        }
    }
    Ok(last)
}

/// The latest live model to fall back to when a requested model id 404s.
fn anthropic_fallback_model(model: &str) -> Option<&'static str> {
    match model {
        "claude-sonnet-4-7" => Some("claude-sonnet-4-6"),
        "claude-haiku-4-7" => Some("claude-haiku-4-5-20251001"),
        _ => None,
    }
}

/// Send one Anthropic `/v1/messages` request for `model`; returns the HTTP status and JSON body.
fn anthropic_send(
    client: &Client,
    settings: &Settings,
    model: &str,
    system_prompt: &str,
    msgs: &[serde_json::Value],
) -> Result<(reqwest::StatusCode, serde_json::Value)> {
    let oauth = settings
        .anthropic_oauth_token
        .as_ref()
        .filter(|t| !t.trim().is_empty());

    // Coding tasks emit whole files; a small cap truncates them. 4.x allow large outputs.
    let max_tokens = if model.contains("sonnet-4") || model.contains("opus-4") {
        32000
    } else if model.contains("-4-") {
        16384
    } else {
        8192
    };
    let messages_val = serde_json::Value::Array(msgs.to_vec());

    // Prefer the migrated Claude Code OAuth token (Bearer + oauth beta + Claude Code system
    // identity); otherwise use the x-api-key path.
    let req = client.post("https://api.anthropic.com/v1/messages");
    let (req, body) = if let Some(token) = oauth {
        let body = json!({
            "model": model,
            "max_tokens": max_tokens,
            "system": [
                {"type": "text", "text": "You are Claude Code, Anthropic's official CLI for Claude."},
                {"type": "text", "text": system_prompt}
            ],
            "messages": messages_val
        });
        let req = req
            .header("authorization", format!("Bearer {token}"))
            .header("anthropic-version", "2023-06-01")
            .header("anthropic-beta", "oauth-2025-04-20");
        (req, body)
    } else {
        let key = settings
            .anthropic_api_key
            .clone()
            .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
            .context("Anthropic credentials missing (set an API key or run /migration claude)")?;
        let body = json!({
            "model": model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": messages_val
        });
        let req = req
            .header("x-api-key", key)
            .header("anthropic-version", "2023-06-01");
        (req, body)
    };

    let resp = req.json(&body).send().context("Anthropic request failed")?;
    let status = resp.status();
    let v: serde_json::Value = resp.json().context("Anthropic returned invalid JSON")?;
    Ok((status, v))
}

fn call_glm_bigmodel(
    client: &Client,
    settings: &Settings,
    model: &str,
    messages: &[serde_json::Value],
) -> Result<String> {
    let key = settings
        .glm_api_key
        .clone()
        .or_else(|| std::env::var("ZHIPU_API_KEY").ok())
        .or_else(|| std::env::var("GLM_API_KEY").ok())
        .context("GLM / ZHIPU_API_KEY missing")?;
    let body = json!({"model": model, "messages": messages});
    let resp = client
        .post("https://open.bigmodel.cn/api/paas/v4/chat/completions")
        .bearer_auth(key)
        .json(&body)
        .send()
        .context("GLM BigModel request failed")?;
    let status = resp.status();
    let v: serde_json::Value = resp
        .json()
        .context("GLM BigModel returned invalid JSON")?;
    if !status.is_success() {
        let err = v["error"]["message"]
            .as_str()
            .or_else(|| v["message"].as_str())
            .unwrap_or("Unknown GLM error");
        return Ok(format!("GLM API error ({status}): {err}"));
    }
    extract_openai_like_content("GLM", &v)
}

fn call_moonshot(
    client: &Client,
    settings: &Settings,
    model: &str,
    messages: &[serde_json::Value],
) -> Result<String> {
    let key = settings
        .moonshot_api_key
        .clone()
        .or_else(|| std::env::var("MOONSHOT_API_KEY").ok())
        .context("MOONSHOT_API_KEY missing")?;
    let body = json!({"model": model, "messages": messages});
    let v: serde_json::Value = client
        .post("https://api.moonshot.ai/v1/chat/completions")
        .bearer_auth(key)
        .json(&body)
        .send()
        .context("Moonshot request failed")?
        .json()
        .context("Moonshot returned invalid JSON")?;
    extract_openai_like_content("Moonshot", &v)
}

fn call_copilot(
    client: &Client,
    settings: &Settings,
    model: &str,
    messages: &[serde_json::Value],
) -> Result<String> {
    let key = settings
        .github_copilot_token
        .clone()
        .or_else(|| std::env::var("GITHUB_COPILOT_TOKEN").ok())
        .context("GITHUB_COPILOT_TOKEN missing")?;
    let body = json!({"model": model, "messages": messages});
    let v: serde_json::Value = client
        .post("https://api.githubcopilot.com/chat/completions")
        .bearer_auth(key)
        .json(&body)
        .send()
        .context("GitHub Copilot request failed")?
        .json()
        .context("GitHub Copilot returned invalid JSON")?;
    extract_openai_like_content("GitHub Copilot", &v)
}

fn extract_openai_like_content(provider: &str, v: &serde_json::Value) -> Result<String> {
    if let Some(content) = v["choices"][0]["message"]["content"].as_str() {
        if !content.trim().is_empty() {
            return Ok(content.to_string());
        }
    }
    if let Some(err) = v["error"]["message"].as_str() {
        return Ok(format!("{provider} API error: {err}"));
    }
    Ok(format!(
        "{provider} returned no content. Check model name and API key scopes."
    ))
}
