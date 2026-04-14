use anyhow::{Context, Result};
use reqwest::blocking::Client;
use serde_json::json;

use crate::ai::sandbox::run_in_sandbox;
use crate::core::models::{Message, Provider, Settings};

pub fn validate_settings_keys(settings: &Settings) -> Vec<String> {
    vec![
        validate_one("OpenAI", settings.openai_api_key.as_deref(), &["sk-"]),
        validate_one("Groq", settings.groq_api_key.as_deref(), &["gsk_"]),
        validate_one("Anthropic", settings.anthropic_api_key.as_deref(), &["sk-ant-"]),
        validate_one("Moonshot", settings.moonshot_api_key.as_deref(), &["sk-", "moon-"]),
        validate_one(
            "GitHub Copilot",
            settings.github_copilot_token.as_deref(),
            &["ghu_", "github_pat_", "ghp_"],
        ),
    ]
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
    for m in history.iter().rev().take(12).rev() {
        messages.push(json!({"role": m.role, "content": m.content}));
    }

    let client = Client::new();
    let assignment = select_provider_for_subagent(settings, subagent);
    let answer = match assignment.provider {
        Provider::OpenAi => call_openai(&client, settings, &assignment.model, &messages)?,
        Provider::Groq => call_groq(&client, settings, &assignment.model, &messages)?,
        Provider::Anthropic => {
            call_anthropic(&client, settings, &assignment.model, history, &system_prompt)?
        }
        Provider::MoonshotAi => call_moonshot(&client, settings, &assignment.model, &messages)?,
        Provider::GithubCopilot => call_copilot(&client, settings, &assignment.model, &messages)?,
    };
    Ok(format!("[{}] {}", settings.ai_avatar, answer))
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
Speak naturally and briefly to the user about what you changed. \
When you want files written, include machine-readable blocks exactly like:
<OPENCAGE_FILE path=\"relative/path.ext\">
...full file content...
</OPENCAGE_FILE>
When you want terminal commands run, include blocks like:
<OPENCAGE_CMD>command here</OPENCAGE_CMD>
You may include multiple file/cmd blocks. {depth_hint} {detail_hint}"
    );
    let mut messages = vec![json!({"role":"system","content":system_prompt})];
    for m in history.iter().rev().take(10).rev() {
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
    let mut enabled = if settings.enabled_providers.is_empty() {
        vec![settings.provider.clone()]
    } else {
        settings.enabled_providers.clone()
    };
    enabled.retain(|p| provider_has_credentials(settings, p));
    if enabled.is_empty() && provider_has_credentials(settings, &settings.provider) {
        enabled.push(settings.provider.clone());
    }
    if enabled.is_empty() {
        enabled = Provider::all()
            .into_iter()
            .filter(|p| provider_has_credentials(settings, p))
            .collect();
    }
    enabled
}

fn preferred_order(subagent: SubAgent) -> Vec<Provider> {
    match subagent {
        SubAgent::Research => vec![Provider::Anthropic, Provider::OpenAi, Provider::MoonshotAi, Provider::Groq, Provider::GithubCopilot],
        SubAgent::Reviewer => vec![Provider::Anthropic, Provider::OpenAi, Provider::Groq, Provider::MoonshotAi, Provider::GithubCopilot],
        SubAgent::Coding => vec![Provider::OpenAi, Provider::Groq, Provider::GithubCopilot, Provider::Anthropic, Provider::MoonshotAi],
        SubAgent::Shell => vec![Provider::OpenAi, Provider::Groq, Provider::Anthropic, Provider::GithubCopilot, Provider::MoonshotAi],
        SubAgent::General => vec![Provider::OpenAi, Provider::Anthropic, Provider::Groq, Provider::MoonshotAi, Provider::GithubCopilot],
    }
}

fn provider_has_credentials(settings: &Settings, provider: &Provider) -> bool {
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
            || std::env::var("ANTHROPIC_API_KEY").ok().is_some(),
        Provider::MoonshotAi => settings
            .moonshot_api_key
            .as_ref()
            .is_some_and(|v| !v.trim().is_empty())
            || std::env::var("MOONSHOT_API_KEY").ok().is_some(),
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
    let key = settings
        .anthropic_api_key
        .clone()
        .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
        .context("ANTHROPIC_API_KEY missing")?;
    let anthropic_msgs: Vec<_> = history
        .iter()
        .map(|m| {
            json!({
                "role": if m.role == "assistant" { "assistant" } else { "user" },
                "content": m.content
            })
        })
        .collect();
    let body = json!({
        "model": model,
        "max_tokens": 1024,
        "system": system_prompt,
        "messages": anthropic_msgs
    });
    let v: serde_json::Value = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", key)
        .header("anthropic-version", "2023-06-01")
        .json(&body)
        .send()
        .context("Anthropic request failed")?
        .json()
        .context("Anthropic returned invalid JSON")?;
    Ok(v["content"][0]["text"]
        .as_str()
        .unwrap_or("No content")
        .to_string())
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
