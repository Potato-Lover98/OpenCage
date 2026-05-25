# OpenCage

A fast, private **terminal AI assistant** — a TUI built in Rust on top of
[`ratatui`](https://github.com/ratatui/ratatui) and `crossterm`. Chat with multiple
AI providers, keep long‑term memory, talk to it by voice, paste images, resume past
sessions, and let it write code — all without leaving your terminal.

> Your API keys are stored **encrypted on disk** and never leave your machine except to
> reach the provider you choose.

## Features

- **Multi‑provider chat** — Groq, OpenAI, Anthropic, Moonshot, GLM (BigModel), and GitHub
  Copilot. Switch the active provider/model at any time.
- **Encrypted key storage** — API keys are kept in an embedded, AES‑256‑GCM‑encrypted
  store; the master key comes from `OPENCAGE_MASTER_KEY` or a local key file.
- **Long‑term memory (RAG)** — remember facts across sessions and have them recalled as
  context automatically.
- **Voice input** *(optional)* — push‑to‑talk transcription via Whisper.
- **Image paste** *(optional)* — paste an image from the clipboard into the conversation.
- **Sessions** — conversations are saved and can be resumed later.
- **Autonomous coding** — ask OpenCage to build or edit code; it writes files into your
  project and proposes shell commands for your approval (untrusted folders ask first).
- **Command palette** — type `/` to discover and run slash commands.

## Quick start

```bash
# 1. Build
cargo build --release

# 2. Run
cargo run --release
```

On first launch, open the settings tab with `/settings`, paste an API key for at least one
provider, press **Ctrl+S** to save, then start chatting. See **[SETUP.md](SETUP.md)** for
prerequisites, feature flags, environment variables, and cross‑compilation.

## Common commands

Type `/` in the input to open the palette. A few of the most useful:

| Command | What it does |
| --- | --- |
| `/settings` | Open settings — providers, models, API keys, options |
| `/model` | Show (or switch) the active provider and model |
| `/sessions` | Browse and resume saved conversations |
| `/new` | Start a fresh session |
| `/voice` | Toggle push‑to‑talk voice input |
| `/remember <fact>` | Save a fact to long‑term memory |
| `/memories [query]` | Recall saved memories |
| `/deep [on\|off\|0-10]` | Adjust reasoning depth |
| `/blacklist` | Manage commands the agent may never run |
| `/help` | List all commands |

## Configuration

- **`OPENCAGE_MASTER_KEY`** (recommended): base64 of 32 bytes; decrypts your stored API
  keys at runtime. Without it, a key file is generated under your config directory.
- Settings and encrypted keys live in your OS config dir (`…/opencage/`); sessions and
  memory live in `~/.opencage/`.

See **[SETUP.md](SETUP.md)** for the full list of environment variables and storage paths.

## Building notes

- Rust **2024 edition**.
- Default features enable voice (`local_voice`) and clipboard image paste
  (`native_clipboard`). Build without them using `--no-default-features`.
- Cross‑compiling Linux → macOS uses `cargo zigbuild --no-default-features` (see SETUP.md).

## License

[MIT](LICENSE) © 2026 OpenCage contributors.
