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

Runtime wiring (encoder selection by audio length) lands in a follow-up PR.
