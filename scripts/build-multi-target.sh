#!/usr/bin/env bash
set -euo pipefail

APP_NAME="opencage"
TARGETS=(
  "x86_64-unknown-linux-gnu"
  "aarch64-unknown-linux-gnu"
  "x86_64-apple-darwin"
  "aarch64-apple-darwin"
  "x86_64-pc-windows-gnu"
  "aarch64-pc-windows-gnullvm"
)

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

need_cmd cargo
need_cmd rustup
need_cmd zig

if ! cargo zigbuild --version >/dev/null 2>&1; then
  echo "Installing cargo-zigbuild..."
  cargo install cargo-zigbuild
fi

for target in "${TARGETS[@]}"; do
  echo "==> Building target: $target"
  rustup target add "$target" || true
  cargo zigbuild --release --target "$target"
done

DIST_DIR="dist"
mkdir -p "$DIST_DIR"

for target in "${TARGETS[@]}"; do
  src="target/$target/release/$APP_NAME"
  out="$DIST_DIR/${APP_NAME}-${target}"
  if [[ "$target" == *"windows"* ]]; then
    src="${src}.exe"
    out="${out}.exe"
  fi
  if [[ -f "$src" ]]; then
    cp "$src" "$out"
    echo "Created: $out"
  else
    echo "Warning: build output missing for $target at $src" >&2
  fi
done

echo
echo "Done. Binaries are in ./$DIST_DIR"
