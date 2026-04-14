# OpenCage

OpenCage is a Rust terminal AI agent with a colorful TUI, file tree, chat workflow, provider-backed LLM support, long-term memory, and autonomous coding with user safety prompts.

## Features

- Split-pane TUI chat + file tree built with `ratatui` and `crossterm`
- Multi-provider support (Groq, OpenAI, Anthropic, Moonshot AI, GitHub Copilot)
- Sub-agent routing for coding, research, review, and shell-focused tasks
- RAG-style memory using `heed`
- Encrypted settings/key storage
- Folder trust and per-command approval for autonomous execution
- Slash command palette (settings, memory, blacklist, deep-think controls, and more)
- Cross-platform build script for Linux, macOS, and Windows targets

## Build

```bash
cargo build --release
```

Binary output:

```bash
target/release/opencage
```

## Run

```bash
./target/release/opencage
```

## Cross Compilation

See `BUILD_CROSS.md` and run:

```bash
bash scripts/build-multi-target.sh
```

## Safety Model

- Command execution uses explicit yes/no prompts
- Folder trust is required before autonomous coding writes
- Blacklisted commands are blocked and can be edited via `/blacklist`

## License

This project is licensed under the MIT License. See `LICENSE`.
