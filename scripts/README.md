# VoiceInk scripts

Tooling that lives outside the app binary. Not needed for regular development — only for generating model artifacts or running offline profiling / export flows.

## `export-encoder-shapes.py`

Exports the whisper encoder at multiple fixed input lengths (5s / 10s / 15s / 30s) as CoreML models. Used for variable-latency ANE inference — see the inline docstring for rationale.

Requires [uv](https://docs.astral.sh/uv/) (dependencies are declared inline in the script via [PEP 723](https://peps.python.org/pep-0723/)):

```bash
uv run scripts/export-encoder-shapes.py \
    --model large-v3-turbo \
    --output-dir ~/VoiceInk-Encoders
```

Outputs both `.mlpackage` (intermediate) and `.mlmodelc` (runtime-ready, if `xcrun` is available) for each shape. The `.mlmodelc` files are what the app loads at runtime.

## `install-encoder-variants.sh`

One-shot wrapper that exports the variants and drops them directly into VoiceInk's on-disk models directory (`~/Library/Application Support/com.prakashjoshipax.VoiceInk/WhisperModels/`). Prerequisite: the target model (`large-v3-turbo` by default) must already be downloaded via VoiceInk's model manager — variants are siblings of the stock encoder.

```bash
scripts/install-encoder-variants.sh              # defaults to large-v3-turbo
scripts/install-encoder-variants.sh large-v2     # or any other downloaded model
```

Override the destination via `VOICEINK_MODELS_DIR=/path/to/WhisperModels` if the bundle id or sandbox path differs.

After install, relaunch VoiceInk. The xcframework's `whisper-coreml` layer probes for sibling `-5s` / `-10s` / `-15s` / `-30s` `.mlmodelc` files automatically and logs `loaded N shape variant(s)` on successful dispatch.
