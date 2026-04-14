#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = [
#     "torch==2.5.0",
#     "coremltools>=8.0",
#     "openai-whisper @ git+https://github.com/openai/whisper.git",
#     "numpy<2",
# ]
# ///
"""Export whisper encoders at multiple fixed input lengths for variable-latency ANE inference.

The stock whisper.cpp CoreML encoder is hardcoded to 3000 mel frames (30s). For a typical 3s
dictation this wastes ~85% of encoder compute padding silence. Exporting multiple fixed-shape
encoders (5s/10s/15s/30s) and dispatching at runtime to the smallest one that fits yields
a ~5-10x speedup on short utterances — which is most of our traffic.

Why fixed shapes instead of a single dynamic-shape model:
    CoreML does support ct.RangeDim, but ANE compilation can silently fall back to GPU for
    unseen shapes, producing unpredictable regressions. Fixed shapes guarantee ANE execution.

Usage:
    uv run scripts/export-encoder-shapes.py \\
        --model large-v3-turbo \\
        --output-dir ~/VoiceInk-Encoders

Output (per shape):
    <output-dir>/ggml-large-v3-turbo-encoder-5s.mlpackage
    <output-dir>/ggml-large-v3-turbo-encoder-5s.mlmodelc    (compiled, if xcrun available)
    ...

The generated .mlmodelc files are what VoiceInk ships at runtime. The .mlpackage is the
intermediate artifact and can be deleted after compilation.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import coremltools as ct
import torch
import torch.nn.functional as F

# MultiHeadAttention.use_sdpa is a class attribute checked at forward-time,
# so toggling it before load_model() is sufficient to avoid torch 2.5 SDPA
# tracing issues. This mirrors whisper.cpp/models/convert-whisper-to-coreml.py.
import whisper.model  # noqa: E402

whisper.model.MultiHeadAttention.use_sdpa = False

from whisper import load_model  # noqa: E402
from whisper.model import AudioEncoder  # noqa: E402

# Mel frames -> audio duration:
#   whisper processes audio at 16kHz sample rate and computes 100 mel frames per second.
#   After the stride-2 conv2 in the encoder stem, the transformer context length is
#   n_mel_frames / 2. For whisper-large-v3-turbo the stock 30s input is 3000 mel frames
#   and produces a 1500-token encoder memory.
SHAPES: dict[str, int] = {
    "5s": 500,
    "10s": 1000,
    "15s": 1500,
    "30s": 3000,
}


class SlicedEncoder(torch.nn.Module):
    """Wraps a whisper AudioEncoder to accept shorter mel inputs.

    The only difference vs. the stock encoder is that positional_embedding is sliced
    to match the post-conv time dimension, rather than asserting the input is exactly
    30s (3000 frames -> 1500 ctx). The underlying weights are unchanged — this is a
    pure forward-pass variant.

    Torch JIT trace captures the slice at a fixed length, so one instance of this class
    corresponds to one exported fixed-shape encoder.
    """

    def __init__(self, encoder: AudioEncoder, n_mel_frames: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.n_audio_ctx = n_mel_frames // 2

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.encoder.conv1(mel))
        x = F.gelu(self.encoder.conv2(x))
        x = x.permute(0, 2, 1)  # (B, n_audio_ctx, n_state)
        pe = self.encoder.positional_embedding[: self.n_audio_ctx]
        x = (x + pe).to(x.dtype)
        for block in self.encoder.blocks:
            x = block(x)
        x = self.encoder.ln_post(x)
        return x


def convert_shape(
    whisper_model,
    n_mel_frames: int,
) -> ct.models.MLModel:
    # SlicedEncoder relies on n_mel_frames // 2 matching the conv2 output time
    # dimension. That arithmetic only holds for even n_mel_frames (conv2 is
    # kernel=3, stride=2, padding=1 -> floor((N-1)/2)+1, which equals N/2
    # iff N is even). All four shapes in SHAPES are even; guard the
    # assumption so a future addition fails loudly instead of producing a
    # positional-embedding shape mismatch at trace time.
    assert n_mel_frames % 2 == 0, f"n_mel_frames must be even, got {n_mel_frames}"

    encoder = SlicedEncoder(whisper_model.encoder, n_mel_frames).eval()
    n_mels = whisper_model.dims.n_mels
    input_shape = (1, n_mels, n_mel_frames)
    example = torch.randn(*input_shape)

    with torch.no_grad():
        traced = torch.jit.trace(encoder, example)

    # mlprogram defaults to compute_precision=FLOAT16, which stores weights
    # as FP16 — ~half the disk footprint of FP32 and what ANE expects anyway.
    # Being explicit here makes that contract visible rather than relying on
    # the coremltools default.
    return ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="logmel_data", shape=input_shape)],
        outputs=[ct.TensorType(name="output")],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS14,
    )


def compile_mlmodelc(mlpackage: Path) -> Path | None:
    """Compile an .mlpackage to the runtime .mlmodelc using xcrun.

    Returns the path to the compiled .mlmodelc, or None if xcrun isn't available
    (which is fine on non-macOS dev machines — the .mlpackage can still be shipped
    and compiled on the target device).
    """
    if shutil.which("xcrun") is None:
        print("  xcrun not available; skipping .mlmodelc compile")
        return None

    output_dir = mlpackage.parent
    subprocess.check_call(
        [
            "xcrun",
            "coremlcompiler",
            "compile",
            str(mlpackage),
            str(output_dir),
        ]
    )
    produced = output_dir / (mlpackage.stem + ".mlmodelc")
    if not produced.exists():
        raise RuntimeError(f"coremlcompiler did not produce {produced}")
    return produced


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="large-v3-turbo",
        help="whisper model id (default: large-v3-turbo)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory to write .mlpackage / .mlmodelc outputs",
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=list(SHAPES.keys()),
        choices=list(SHAPES.keys()),
        help="which window sizes to export (default: all)",
    )
    parser.add_argument(
        "--no-compile",
        dest="compile",
        action="store_false",
        default=True,
        help="skip the xcrun coremlcompiler step (keep only .mlpackage)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading whisper model: {args.model}")
    model = load_model(args.model).cpu().eval()
    dims = model.dims
    print(
        f"  n_mels={dims.n_mels}, n_audio_ctx={dims.n_audio_ctx}, "
        f"n_audio_state={dims.n_audio_state}, n_audio_layer={dims.n_audio_layer}"
    )

    for shape_name in args.shapes:
        n_mel = SHAPES[shape_name]
        print(f"\nExporting {shape_name} encoder ({n_mel} mel frames -> {n_mel // 2} audio ctx)")
        ct_model = convert_shape(model, n_mel)

        mlpackage = args.output_dir / f"ggml-{args.model}-encoder-{shape_name}.mlpackage"
        ct_model.save(str(mlpackage))
        print(f"  wrote {mlpackage}")

        if args.compile:
            mlmodelc = compile_mlmodelc(mlpackage)
            if mlmodelc is not None:
                print(f"  compiled -> {mlmodelc}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
