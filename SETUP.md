# Setup

How to build, configure, and run OpenCage.

## 1. Prerequisites

- **Rust** (2024 edition) — install via [rustup](https://rustup.rs):
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```
- A **C/C++ toolchain** and **CMake** (needed by some native dependencies).

### Linux system packages (for the default features)

The default build enables voice (`cpal` + `whisper-rs`) and clipboard image paste
(`arboard`). On Debian/Ubuntu:

```bash
sudo apt update
sudo apt install build-essential cmake pkg-config libasound2-dev libxcb1-dev
```

(Package names vary by distro — install the ALSA dev headers and X11/XCB clipboard libs
for your system. If you don't need voice or clipboard, skip these and build with
`--no-default-features`, below.)

## 2. Build & run

```bash
# Debug build
cargo build
cargo run

# Optimized build (recommended for daily use)
cargo build --release
cargo run --release
```

### Feature flags

| Feature | Default | Enables |
| --- | --- | --- |
| `local_voice` | ✅ | Microphone capture (`cpal`) + offline Whisper transcription |
| `native_clipboard` | ✅ | Paste images from the system clipboard (`arboard`) |

Build a lean binary without them:

```bash
cargo build --release --no-default-features
```

### Cross‑compiling Linux → macOS

The default features link Apple frameworks that Zig can't resolve on Linux, so disable
them:

```bash
cargo zigbuild --release --no-default-features --target aarch64-apple-darwin
```

## 3. Configuration

### Master key (encrypts your API keys)

OpenCage stores provider API keys **encrypted at rest**. Provide the 32‑byte master key as
base64 via an environment variable (recommended), or let OpenCage generate and save a key
file on first run.

```bash
# Generate a key and export it (keep this safe — losing it means re-entering your API keys)
export OPENCAGE_MASTER_KEY="$(head -c 32 /dev/urandom | base64)"
```

Add that `export` to your shell profile (`~/.bashrc`, `~/.zshrc`, …) so it persists.

### Adding provider API keys

1. Run OpenCage.
2. Open settings: type `/settings`.
3. Select a provider field, paste its API key.
4. Press **Ctrl+S** to save (keys are encrypted with your master key).
5. Press **F5** to validate the configured keys.

Supported providers: Groq, OpenAI, Anthropic, Moonshot, GLM (BigModel), GitHub Copilot.

### Environment variables

| Variable | Purpose |
| --- | --- |
| `OPENCAGE_MASTER_KEY` | Base64 of 32 bytes; decrypts stored API keys at runtime |
| `OPENCAGE_WHISPER_MODEL` | Path to a local Whisper model for offline voice transcription |
| `OPENCAGE_BLACKLIST_EDITOR` | Editor to open the command blacklist file |
| `EDITOR` | Fallback editor for external editing |

Provider keys may also be supplied via the usual environment variables
(`OPENAI_API_KEY`, `GROQ_API_KEY`, `ANTHROPIC_API_KEY`, `MOONSHOT_API_KEY`,
`GLM_API_KEY` / `ZHIPU_API_KEY`, `GITHUB_COPILOT_TOKEN`) if you prefer not to store them.

## 4. Where data lives

| Data | Location |
| --- | --- |
| Encrypted settings + API keys | `…/opencage/settings_db` (OS config dir) |
| Master key file (if not using the env var) | `…/opencage/settings.key` |
| Saved sessions | `~/.opencage/sessions/` |
| Long‑term memory (RAG) | `~/.opencage/rag_db/` |

## 5. Troubleshooting

- **"decryption failed (wrong key or corrupted data)"** — `OPENCAGE_MASTER_KEY` doesn't
  match the key your settings were encrypted with. Use the original key, or delete the
  settings store and re‑enter your API keys.
- **Build fails on `cpal` / `whisper-rs` / `arboard`** — install the system packages above,
  or build with `--no-default-features`.
- **Keys "missing" despite being set** — make sure you pressed **Ctrl+S** in `/settings`,
  and that `OPENCAGE_MASTER_KEY` is exported in the shell you launch OpenCage from.
