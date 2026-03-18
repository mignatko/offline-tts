#!/usr/bin/env python3
"""
txt_to_hu_mp3_mms.py

Convert a UTF-8 Hungarian Q/A text file to MP3 using a local MMS Hungarian TTS model.

Expected input format:

Q
A

Q
A

Q
A

Non-empty lines are treated as alternating:
1st = question
2nd = answer
3rd = question
4th = answer
...

Requirements:
    pip install torch transformers soundfile
    ffmpeg must be installed and available in PATH

Example:
    python txt_to_hu_mp3_mms.py \
        --input input.txt \
        --output output.mp3 \
        --model-dir models/mms-tts-hun \
        --pause 1.0 \
        --question-repeats 2 \
        --answer-repeats 3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoTokenizer, VitsModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Hungarian Q/A text file to MP3 using local MMS Hungarian TTS."
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="Path to the input .txt file (UTF-8).",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Path to the output .mp3 file.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/mms-tts-hun"),
        help="Path to the local MMS Hungarian model directory.",
    )

    parser.add_argument(
        "-p",
        "--pause",
        type=float,
        default=1.0,
        help="Pause duration in seconds between spoken items.",
    )

    parser.add_argument(
        "--question-repeats",
        type=int,
        default=1,
        help="How many times to repeat each question line. Default: 1",
    )
    parser.add_argument(
        "--answer-repeats",
        type=int,
        default=1,
        help="How many times to repeat each answer line. Default: 1",
    )

    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cpu"],
        default="auto",
        help="Inference device. Default: auto",
    )

    parser.add_argument(
        "--speaking-rate",
        type=float,
        default=1.0,
        help="Approximate speaking rate multiplier. 1.0 = original. "
             "Values > 1.0 speed up, values < 1.0 slow down.",
    )

    return parser.parse_args()


def print_step(message: str) -> None:
    print(f"[INFO] {message}")


def print_progress(current: int, total: int, prefix: str = "Progress") -> None:
    if total <= 0:
        return

    percent = current / total * 100
    bar_width = 30
    filled = int(bar_width * current / total)
    bar = "#" * filled + "-" * (bar_width - filled)

    sys.stdout.write(f"\r{prefix}: [{bar}] {current}/{total} ({percent:5.1f}%)")
    sys.stdout.flush()

    if current == total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def validate_args(args: argparse.Namespace) -> None:
    if args.pause < 0:
        raise ValueError("--pause must be >= 0")
    if args.question_repeats < 1:
        raise ValueError("--question-repeats must be >= 1")
    if args.answer_repeats < 1:
        raise ValueError("--answer-repeats must be >= 1")
    if args.speaking_rate <= 0:
        raise ValueError("--speaking-rate must be > 0")
    if not args.input.exists():
        raise FileNotFoundError(f"Input file does not exist: {args.input}")
    if not args.input.is_file():
        raise ValueError(f"Input path is not a file: {args.input}")
    if not args.model_dir.exists():
        raise FileNotFoundError(
            f"Model directory does not exist: {args.model_dir}\n"
            f"Download the model first."
        )
    if not args.model_dir.is_dir():
        raise ValueError(f"Model path is not a directory: {args.model_dir}")


def ensure_ffmpeg() -> None:
    from shutil import which

    if which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found in PATH. Please install ffmpeg first.")


def read_input_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def resolve_device(device_arg: str) -> str:
    if device_arg == "cpu":
        return "cpu"
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return "mps"

    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_dir: Path, device: str) -> tuple[AutoTokenizer, VitsModel]:
    print_step(f"Loading tokenizer from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
    )

    print_step(f"Loading model from: {model_dir}")
    model = VitsModel.from_pretrained(
        model_dir,
        local_files_only=True,
    ).to(device)

    model.eval()
    return tokenizer, model


def synthesize_wav(
    text: str,
    tokenizer: AutoTokenizer,
    model: VitsModel,
    device: str,
    output_path: Path,
) -> int:
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        waveform = model(**inputs).waveform

    audio = waveform.cpu().float().numpy().squeeze()
    sample_rate = int(model.config.sampling_rate)
    sf.write(str(output_path), audio, sample_rate)
    return sample_rate


def generate_silence_wav(duration_seconds: float, sample_rate: int, output_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r={sample_rate}:cl=mono",
            "-t",
            str(duration_seconds),
            str(output_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def apply_tempo_filter(input_wav: Path, output_wav: Path, speaking_rate: float) -> None:
    if abs(speaking_rate - 1.0) < 1e-9:
        input_wav.replace(output_wav)
        return

    # ffmpeg atempo supports 0.5..2.0 per filter, so chain if needed
    factors: list[float] = []
    remaining = speaking_rate

    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0

    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5

    factors.append(remaining)
    atempo_chain = ",".join(f"atempo={factor:.6f}" for factor in factors)

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_wav),
            "-filter:a",
            atempo_chain,
            str(output_wav),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def concat_to_mp3(parts: list[Path], output_path: Path) -> None:
    print_step("Merging audio parts into final MP3...")

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".txt",
        delete=False,
    ) as f:
        concat_list = Path(f.name)
        for part in parts:
            f.write(f"file '{part.as_posix()}'\n")

    try:
        print_progress(1, 1, prefix="Merging")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
                "-c:a",
                "libmp3lame",
                "-q:a",
                "2",
                str(output_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    finally:
        concat_list.unlink(missing_ok=True)


def prepare_lines(input_text: str) -> list[str]:
    return [line.strip() for line in input_text.split("\n") if line.strip()]


async def main() -> int:
    args = parse_args()

    try:
        started_at = time.perf_counter()

        validate_args(args)
        ensure_ffmpeg()

        device = resolve_device(args.device)
        print_step(f"Using device: {device}")

        print_step(f"Reading input file: {args.input}")
        input_text = read_input_text(args.input)
        if input_text == "":
            raise ValueError("Input file is empty.")

        lines = prepare_lines(input_text)
        if not lines:
            raise ValueError("Input file contains no non-empty text lines to synthesize.")

        tokenizer, model = load_model(args.model_dir, device)

        total_output_segments = 0
        for idx, _ in enumerate(lines, start=1):
            is_question = idx % 2 == 1
            total_output_segments += (
                args.question_repeats if is_question else args.answer_repeats
            )

        print_step(f"Pause between spoken items: {args.pause:.2f}s")
        print_step(f"Question repeats: {args.question_repeats}")
        print_step(f"Answer repeats: {args.answer_repeats}")
        print_step(f"Speaking rate: {args.speaking_rate:.2f}")
        print_step(f"Input text lines: {len(lines)}")
        print_step(f"Speech segments to generate: {total_output_segments}")

        args.output.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="hu_mms_tts_") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)

            print_step("Generating first speech sample to determine sampling rate...")
            probe_wav = tmp_dir / "probe.wav"
            sample_rate = synthesize_wav(
                text=lines[0],
                tokenizer=tokenizer,
                model=model,
                device=device,
                output_path=probe_wav,
            )

            # Remove probe from final flow; it was just for sample rate discovery.
            probe_wav.unlink(missing_ok=True)

            silence_wav = tmp_dir / "silence.wav"
            print_step(f"Generating silence segment ({args.pause:.2f}s)...")
            generate_silence_wav(args.pause, sample_rate, silence_wav)

            parts: list[Path] = []
            generated_segments = 0

            print_step("Generating speech audio...")

            for idx, text in enumerate(lines, start=1):
                is_question = idx % 2 == 1
                repeats = args.question_repeats if is_question else args.answer_repeats

                for _ in range(repeats):
                    generated_segments += 1

                    raw_wav = tmp_dir / f"speech_raw_{generated_segments:05d}.wav"
                    final_wav = tmp_dir / f"speech_{generated_segments:05d}.wav"

                    synthesize_wav(
                        text=text,
                        tokenizer=tokenizer,
                        model=model,
                        device=device,
                        output_path=raw_wav,
                    )

                    apply_tempo_filter(
                        input_wav=raw_wav,
                        output_wav=final_wav,
                        speaking_rate=args.speaking_rate,
                    )

                    if raw_wav.exists():
                        raw_wav.unlink()

                    parts.append(final_wav)

                    is_last_generated_segment = generated_segments == total_output_segments
                    if not is_last_generated_segment:
                        parts.append(silence_wav)

                    print_progress(
                        generated_segments,
                        total_output_segments,
                        prefix="Synthesizing",
                    )

            concat_to_mp3(parts, args.output)

        elapsed = time.perf_counter() - started_at
        size_mb = args.output.stat().st_size / (1024 * 1024)

        print_step(f"Done: {args.output}")
        print_step(f"Output size: {size_mb:.2f} MB")
        print_step(f"Elapsed time: {elapsed:.2f}s")
        return 0

    except subprocess.CalledProcessError:
        print("[ERROR] ffmpeg command failed.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(__import__("asyncio").run(main()))