use std::cmp::Reverse;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use heed::types::{Bytes, Str};
use heed::{Database, Env, EnvOpenOptions};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MemoryEntry {
    scope: String,
    text: String,
    role: String,
    ts: u64,
}

#[derive(Clone)]
pub struct RagStore {
    env: Env,
    db: Database<Str, Bytes>,
}

impl RagStore {
    pub fn open_default() -> Result<Self> {
        let base = if let Some(home) = dirs::home_dir() {
            home.join(".opencage")
        } else {
            PathBuf::from(".opencage")
        };
        fs::create_dir_all(&base).context("Failed to create ~/.opencage directory")?;
        let db_path = base.join("rag_db");
        fs::create_dir_all(&db_path).context("Failed to create RAG DB directory")?;
        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(20 * 1024 * 1024)
                .max_dbs(5)
                .open(&db_path)
        }
        .context("Failed to open heed environment")?;
        let mut wtxn = env.write_txn()?;
        let db = env.create_database::<Str, Bytes>(&mut wtxn, Some("memories"))?;
        wtxn.commit()?;
        Ok(Self { env, db })
    }

    pub fn remember(&self, role: &str, text: &str) -> Result<()> {
        self.remember_scoped("global", role, text)
    }

    pub fn remember_scoped(&self, scope: &str, role: &str, text: &str) -> Result<()> {
        let ts = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        let key = format!("{ts}-{}", rand_suffix(text));
        let entry = MemoryEntry {
            scope: scope.to_string(),
            text: text.to_string(),
            role: role.to_string(),
            ts,
        };
        let value = serde_json::to_vec(&entry)?;
        let mut wtxn = self.env.write_txn()?;
        self.db.put(&mut wtxn, key.as_str(), value.as_slice())?;
        wtxn.commit()?;
        Ok(())
    }

    pub fn retrieve(&self, query: &str, limit: usize) -> Result<Vec<String>> {
        self.retrieve_scoped("global", query, limit)
    }

    pub fn retrieve_scoped(&self, scope: &str, query: &str, limit: usize) -> Result<Vec<String>> {
        let q_tokens = tokenize(query);
        let rtxn = self.env.read_txn()?;
        let mut scored = Vec::new();
        for item in self.db.iter(&rtxn)? {
            let (_, raw) = item?;
            if let Ok(entry) = serde_json::from_slice::<MemoryEntry>(raw) {
                if entry.scope != scope {
                    continue;
                }
                let score = overlap_score(&q_tokens, &tokenize(&entry.text));
                if score > 0 {
                    scored.push((score, entry.ts, entry.text));
                }
            }
        }
        scored.sort_by_key(|(score, ts, _)| (Reverse(*score), Reverse(*ts)));
        Ok(scored.into_iter().take(limit).map(|(_, _, t)| t).collect())
    }
}

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_string)
        .collect()
}

fn overlap_score(a: &[String], b: &[String]) -> usize {
    a.iter().filter(|t| b.contains(*t)).count()
}

fn rand_suffix(seed: &str) -> String {
    let mut h: u64 = 1469598103934665603;
    for b in seed.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    format!("{:x}", h)
}
