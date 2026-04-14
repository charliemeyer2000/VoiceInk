#!/usr/bin/env bash
#
# Export and install shape-specialised whisper CoreML encoder variants
# (5s / 10s / 15s / 30s) into VoiceInk's on-disk models directory.
#
# Runtime dispatch lives in whisper-encoder.mm, which probes for files named
# <base>-<shape>.mlmodelc as siblings of the stock <base>.mlmodelc. Putting
# compiled variants next to the existing encoder is all that's needed to
# activate variable-latency ANE inference — no app rebuild required (just
# relaunch so WhisperContext reloads the encoder).
#
# Usage:
#     scripts/install-encoder-variants.sh [model-name]
#
# model-name defaults to large-v3-turbo (matching VoiceInk's onboarding default).
# Override the destination via VOICEINK_MODELS_DIR if the bundle id differs
# (e.g. notarized fork builds) or the app is sandboxed to a Container path.

set -euo pipefail

MODEL="${1:-large-v3-turbo}"
DEFAULT_MODELS_DIR="${HOME}/Library/Application Support/com.prakashjoshipax.VoiceInk/WhisperModels"
MODELS_DIR="${VOICEINK_MODELS_DIR:-${DEFAULT_MODELS_DIR}}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPORTER="${SCRIPT_DIR}/export-encoder-shapes.py"

if [[ ! -d "${MODELS_DIR}" ]]; then
    echo "error: models directory not found at:" >&2
    echo "  ${MODELS_DIR}" >&2
    echo "" >&2
    echo "Launch VoiceInk and download ${MODEL} from the model manager first," >&2
    echo "or set VOICEINK_MODELS_DIR to the correct path." >&2
    exit 1
fi

STOCK_ENCODER="${MODELS_DIR}/ggml-${MODEL}-encoder.mlmodelc"
if [[ ! -d "${STOCK_ENCODER}" ]]; then
    echo "error: stock encoder not found:" >&2
    echo "  ${STOCK_ENCODER}" >&2
    echo "" >&2
    echo "Download ${MODEL} via VoiceInk's model manager before installing variants." >&2
    echo "Variants are siblings of the stock encoder — without it they have nothing to dispatch from." >&2
    exit 1
fi

echo "==> Target: ${MODELS_DIR}"
echo "==> Model:  ${MODEL}"
echo "==> Exporting + compiling variants (this takes a few minutes on first run)..."
echo

uv run "${EXPORTER}" \
    --model "${MODEL}" \
    --output-dir "${MODELS_DIR}"

echo
echo "==> Removing intermediate .mlpackage artifacts (keeping only runtime .mlmodelc)"
for shape in 5s 10s 15s 30s; do
    mlpackage="${MODELS_DIR}/ggml-${MODEL}-encoder-${shape}.mlpackage"
    if [[ -d "${mlpackage}" ]]; then
        rm -rf "${mlpackage}"
    fi
done

echo
echo "==> Installed variants:"
for shape in 5s 10s 15s 30s; do
    variant="${MODELS_DIR}/ggml-${MODEL}-encoder-${shape}.mlmodelc"
    if [[ -d "${variant}" ]]; then
        size=$(du -sh "${variant}" | awk '{print $1}')
        echo "    ${variant##*/}  (${size})"
    else
        echo "    MISSING: ${variant##*/}" >&2
    fi
done

echo
echo "Done. Relaunch VoiceInk so WhisperContext reloads the encoder."
echo "Check the app logs for 'whisper-coreml: loaded N shape variant(s)' to confirm dispatch."
