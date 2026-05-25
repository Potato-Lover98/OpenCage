use std::collections::HashSet;
use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Provider {
    Groq,
    OpenAi,
    Anthropic,
    MoonshotAi,
    /// Zhipu 智谱 GLM via open.bigmodel.cn (OpenAI-compatible).
    GlmBigModel,
    GithubCopilot,
}

impl Provider {
    pub fn as_str(&self) -> &'static str {
        match self {
            Provider::Groq => "Groq",
            Provider::OpenAi => "OpenAI",
            Provider::Anthropic => "Anthropic",
            Provider::MoonshotAi => "Moonshot AI",
            Provider::GlmBigModel => "GLM (BigModel)",
            Provider::GithubCopilot => "GitHub Copilot",
        }
    }

    pub fn all() -> [Provider; 6] {
        [
            Provider::Groq,
            Provider::OpenAi,
            Provider::Anthropic,
            Provider::MoonshotAi,
            Provider::GlmBigModel,
            Provider::GithubCopilot,
        ]
    }

    pub fn models(&self) -> &'static [&'static str] {
        match self {
            Provider::Groq => &[
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "qwen-qwq-32b",
            ],
            Provider::OpenAi => &["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1", "o4-mini"],
            Provider::Anthropic => &[
                "claude-3-5-haiku-latest",
                "claude-3-5-sonnet-latest",
                "claude-3-7-sonnet-latest",
                "claude-sonnet-4-7",
                "claude-opus-4-7",
                "claude-haiku-4-7",
                "claude-sonnet-4-6",
                "claude-haiku-4-5-20251001",
            ],
            Provider::MoonshotAi => &["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
            Provider::GlmBigModel => &["glm-4-flash", "glm-4", "glm-4-air", "glm-4-plus"],
            Provider::GithubCopilot => &["gpt-4o-mini", "claude-3.5-sonnet", "o3-mini"],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub provider: Provider,
    pub model: String,
    pub ai_avatar: String,
    pub cuss_filter: bool,
    pub groq_api_key: Option<String>,
    pub openai_api_key: Option<String>,
    pub anthropic_api_key: Option<String>,
    pub moonshot_api_key: Option<String>,
    #[serde(default)]
    pub glm_api_key: Option<String>,
    pub github_copilot_token: Option<String>,
    /// OAuth access token migrated from Claude Code (`/migration`). Stored separately from
    /// `anthropic_api_key`; the existing API-key request path is left untouched.
    #[serde(default)]
    pub anthropic_oauth_token: Option<String>,
    /// Unix-seconds expiry for `anthropic_oauth_token`, if known.
    #[serde(default)]
    pub anthropic_oauth_expires_at: Option<u64>,
    /// Plugin ids tracked by OpenCage, mirrored to/from Claude Code (`/plugins`, `/plugout`).
    #[serde(default)]
    pub plugins: Vec<String>,
    /// Telegram bot token for the `/bots` bridge. Stored encrypted; never hard-coded.
    #[serde(default)]
    pub telegram_bot_token: Option<String>,
    pub blocked_commands: HashSet<String>,
    #[serde(default)]
    pub trusted_paths: HashSet<String>,
    #[serde(default)]
    pub enabled_providers: Vec<Provider>,
    #[serde(default)]
    pub provider_models: HashMap<String, String>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            provider: Provider::OpenAi,
            model: "gpt-4o-mini".to_string(),
            ai_avatar: "Opencage".to_string(),
            cuss_filter: true,
            groq_api_key: None,
            openai_api_key: None,
            anthropic_api_key: None,
            moonshot_api_key: None,
            glm_api_key: None,
            github_copilot_token: None,
            anthropic_oauth_token: None,
            anthropic_oauth_expires_at: None,
            plugins: Vec::new(),
            telegram_bot_token: None,
            blocked_commands: ["rm", "shutdown", "reboot", "mkfs", "dd"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            trusted_paths: HashSet::new(),
            enabled_providers: vec![Provider::OpenAi],
            provider_models: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Clone)]
pub struct FileNode {
    pub path: PathBuf,
    pub depth: usize,
    pub is_dir: bool,
    pub expanded: bool,
}
