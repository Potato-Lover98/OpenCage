use std::fs;
use std::path::PathBuf;

use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Nonce};
use anyhow::{Context, Result};
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use heed::types::{Bytes, Str};
use heed::{Database, EnvOpenOptions};
use rand::RngCore;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

use crate::core::models::Settings;

pub struct SettingsStore;

impl SettingsStore {
    pub fn path() -> PathBuf {
        if let Some(config_dir) = dirs::config_dir() {
            config_dir.join("opencage").join("settings_db")
        } else {
            PathBuf::from(".config/opencage/settings_db")
        }
    }

    pub fn load_or_create() -> Result<Settings> {
        let env = Self::open_env()?;
        let mut wtxn = env.write_txn()?;
        let db: Database<Str, Bytes> = env.create_database(&mut wtxn, Some("settings"))?;
        if let Some(raw) = db.get(&wtxn, "active")? {
            let decrypted = decrypt_payload(raw, &Self::load_or_create_key()?)?;
            let cfg: Settings = serde_json::from_slice(&decrypted)
                .context("Invalid encrypted settings payload")?;
            wtxn.commit()?;
            return Ok(cfg);
        }

        if let Some(migrated) = Self::try_migrate_from_legacy_json()? {
            let encrypted = encrypt_payload(&serde_json::to_vec(&migrated)?, &Self::load_or_create_key()?)?;
            db.put(&mut wtxn, "active", encrypted.as_slice())?;
            wtxn.commit()?;
            return Ok(migrated);
        }

        let default = Settings::default();
        let encrypted = encrypt_payload(&serde_json::to_vec(&default)?, &Self::load_or_create_key()?)?;
        db.put(&mut wtxn, "active", encrypted.as_slice())?;
        wtxn.commit()?;
        Ok(default)
    }

    pub fn save(settings: &Settings) -> Result<()> {
        let env = Self::open_env()?;
        let key = Self::load_or_create_key()?;
        let encrypted = encrypt_payload(&serde_json::to_vec(settings)?, &key)?;
        let mut wtxn = env.write_txn()?;
        let db: Database<Str, Bytes> = env.create_database(&mut wtxn, Some("settings"))?;
        db.put(&mut wtxn, "active", encrypted.as_slice())?;
        wtxn.commit()?;
        Ok(())
    }

    fn open_env() -> Result<heed::Env> {
        let path = Self::path();
        fs::create_dir_all(&path)
            .with_context(|| format!("Failed to create settings DB directory {}", path.display()))?;
        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(5 * 1024 * 1024)
                .max_dbs(2)
                .open(&path)
        }
        .with_context(|| format!("Failed to open settings DB at {}", path.display()))?;
        Ok(env)
    }

    fn key_path() -> PathBuf {
        if let Some(config_dir) = dirs::config_dir() {
            config_dir.join("opencage").join("settings.key")
        } else {
            PathBuf::from(".config/opencage/settings.key")
        }
    }

    fn load_or_create_key() -> Result<[u8; 32]> {
        if let Ok(from_env) = std::env::var("OPENCAGE_MASTER_KEY") {
            let bytes = STANDARD
                .decode(from_env.trim())
                .context("OPENCAGE_MASTER_KEY must be base64")?;
            let arr: [u8; 32] = bytes
                .as_slice()
                .try_into()
                .map_err(|_| anyhow::anyhow!("OPENCAGE_MASTER_KEY must decode to exactly 32 bytes"))?;
            return Ok(arr);
        }

        let key_path = Self::key_path();
        if let Some(parent) = key_path.parent() {
            fs::create_dir_all(parent)?;
        }
        if key_path.exists() {
            let text = fs::read_to_string(&key_path)
                .with_context(|| format!("Failed to read key file {}", key_path.display()))?;
            let bytes = STANDARD
                .decode(text.trim())
                .context("Invalid base64 in key file")?;
            let arr: [u8; 32] = bytes
                .as_slice()
                .try_into()
                .map_err(|_| anyhow::anyhow!("Key file must decode to exactly 32 bytes"))?;
            return Ok(arr);
        }

        let mut key = [0_u8; 32];
        rand::thread_rng().fill_bytes(&mut key);
        fs::write(&key_path, STANDARD.encode(key))
            .with_context(|| format!("Failed to write key file {}", key_path.display()))?;
        #[cfg(unix)]
        {
            let perm = fs::Permissions::from_mode(0o600);
            let _ = fs::set_permissions(&key_path, perm);
        }
        Ok(key)
    }

    fn try_migrate_from_legacy_json() -> Result<Option<Settings>> {
        let legacy = if let Some(config_dir) = dirs::config_dir() {
            config_dir.join("opencage").join("settings.json")
        } else {
            PathBuf::from(".config/opencage/settings.json")
        };
        if !legacy.exists() {
            return Ok(None);
        }
        let text = fs::read_to_string(&legacy)
            .with_context(|| format!("Failed to read legacy settings {}", legacy.display()))?;
        let cfg: Settings = serde_json::from_str(&text)
            .with_context(|| format!("Invalid legacy settings format in {}", legacy.display()))?;
        Ok(Some(cfg))
    }
}

fn encrypt_payload(plain: &[u8], key: &[u8; 32]) -> Result<Vec<u8>> {
    let cipher = Aes256Gcm::new_from_slice(key)
        .map_err(|_| anyhow::anyhow!("Invalid encryption key"))?;
    let mut nonce = [0_u8; 12];
    rand::thread_rng().fill_bytes(&mut nonce);
    let ciphertext = cipher
        .encrypt(Nonce::from_slice(&nonce), plain)
        .map_err(|_| anyhow::anyhow!("Encryption failed"))?;
    let mut out = Vec::with_capacity(12 + ciphertext.len());
    out.extend_from_slice(&nonce);
    out.extend_from_slice(&ciphertext);
    Ok(out)
}

fn decrypt_payload(enc: &[u8], key: &[u8; 32]) -> Result<Vec<u8>> {
    if enc.len() < 13 {
        return Err(anyhow::anyhow!("Encrypted payload too small"));
    }
    let cipher = Aes256Gcm::new_from_slice(key)
        .map_err(|_| anyhow::anyhow!("Invalid encryption key"))?;
    let (nonce, data) = enc.split_at(12);
    let plain = cipher
        .decrypt(Nonce::from_slice(nonce), data)
        .map_err(|_| anyhow::anyhow!("Decryption failed (wrong key or corrupted data)"))?;
    Ok(plain)
}
