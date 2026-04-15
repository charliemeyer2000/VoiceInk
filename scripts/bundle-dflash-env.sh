#!/usr/bin/env bash
# Bundle a standalone Python venv with dflash-mlx into the app's Resources.
# Called during the Xcode build phase or manually via `make bundle-dflash`.
#
# Usage: ./scripts/bundle-dflash-env.sh <APP_RESOURCES_DIR>
# Example: ./scripts/bundle-dflash-env.sh .local-build/Build/Products/Debug/VoiceInk.app/Contents/Resources

set -euo pipefail

RESOURCES_DIR="${1:?Usage: $0 <APP_RESOURCES_DIR>}"
ENV_DIR="$RESOURCES_DIR/dflash-env"

if [ -d "$ENV_DIR" ] && [ -f "$ENV_DIR/bin/dflash-serve" ]; then
    echo "dflash-env already exists, skipping"
    exit 0
fi

echo "Creating dflash-env in $ENV_DIR..."

# Use uv to create a standalone venv with dflash-mlx
command -v uv >/dev/null 2>&1 || { echo "uv not found, skipping dflash bundle"; exit 0; }

uv venv "$ENV_DIR" --python 3.13
uv pip install --python "$ENV_DIR/bin/python" "dflash-mlx @ git+https://github.com/bstnxbt/dflash-mlx.git"

echo "dflash-env bundled successfully"
ls -la "$ENV_DIR/bin/dflash-serve"
