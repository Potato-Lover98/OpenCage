# Cross Compilation

This project can produce release binaries for Linux, macOS, and Windows from one machine.

## Prerequisites

- `cargo` and `rustup`
- `zig`

## One-command build

Run:

```bash
bash scripts/build-multi-target.sh
```

It builds:

- `x86_64-unknown-linux-gnu`
- `aarch64-unknown-linux-gnu`
- `x86_64-apple-darwin`
- `aarch64-apple-darwin`
- `x86_64-pc-windows-gnu`
- `aarch64-pc-windows-gnullvm`

Then it copies binaries to `dist/` with target names, for example:

- `dist/opencage-x86_64-unknown-linux-gnu`
- `dist/opencage-aarch64-apple-darwin`
- `dist/opencage-x86_64-pc-windows-gnu.exe`
