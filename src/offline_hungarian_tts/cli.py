from __future__ import annotations

import argparse
import sys
import tempfile
import time
from contextlib import closing
from pathlib import Path
from shutil import which

from .audio import concat_to_mp3, ensure_ffmpeg, read_input_text
from .engines.base import BaseTTSEngine
from .pipeline import build_segment_plan, prepare_lines, render_audio_parts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Hungarian Q/A text file to MP3 using local MMS or Piper."
    )

    parser.add_argument(
        "--engine", choices=["mms", "piper", "xtts"], required=True, help="TTS engine to use."
    )
    parser.add_argument("-i", "--input", required=True, type=Path, help="Input .txt file (UTF-8).")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Output .mp3 file.")
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
        help="How many times to repeat each question line.",
    )
    parser.add_argument(
        "--answer-repeats", type=int, default=1, help="How many times to repeat each answer line."
    )
    parser.add_argument(
        "--speaking-rate",
        type=float,
        default=1.0,
        help="Approximate speaking rate multiplier applied after synthesis.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/mms-tts-hun"),
        help="Local MMS model directory.",
    )
    parser.add_argument(
        "--device", choices=["auto", "mps", "cpu"], default="auto", help="Device for MMS inference."
    )
    parser.add_argument("--piper-model", type=Path, help="Path to Piper .onnx model file.")
    parser.add_argument(
        "--piper-config",
        type=Path,
        help="Path to Piper .onnx.json config file. Defaults to <model>.json",
    )
    parser.add_argument(
        "--piper-bin", default="piper", help="Piper executable name or absolute path."
    )
    parser.add_argument(
        "--speaker", type=int, help="Optional Piper speaker id for multi-speaker voices."
    )
    parser.add_argument("--length-scale", type=float, help="Optional Piper length scale.")
    parser.add_argument("--noise-scale", type=float, help="Optional Piper noise scale.")
    parser.add_argument("--noise-w-scale", type=float, help="Optional Piper noise width scale.")
    parser.add_argument(
        "--xtts-model-dir",
        type=Path,
        default=Path("models/xtts-v2"),
        help="Path to the local XTTS-v2 model directory.",
    )
    parser.add_argument(
        "--xtts-speaker-wav",
        type=Path,
        help="Path to the reference speaker WAV/MP3 file for XTTS voice cloning.",
    )
    parser.add_argument(
        "--xtts-language", default="hu", help="XTTS target language code. For Hungarian use: hu"
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

    if args.engine == "piper" and not args.piper_model:
        raise ValueError("--piper-model is required when --engine piper")

    if (
        args.engine == "piper"
        and which(args.piper_bin) is None
        and not Path(args.piper_bin).exists()
    ):
        raise RuntimeError(f"Piper executable was not found: {args.piper_bin}")

    if args.engine == "xtts" and not args.xtts_speaker_wav:
        raise ValueError("--xtts-speaker-wav is required when --engine xtts")

    if args.engine == "xtts" and not args.xtts_model_dir.exists():
        raise FileNotFoundError(f"XTTS model directory does not exist: {args.xtts_model_dir}")

    if args.engine == "xtts" and not args.xtts_speaker_wav.exists():
        raise FileNotFoundError(
            f"XTTS speaker reference file does not exist: {args.xtts_speaker_wav}"
        )


def create_engine(args: argparse.Namespace) -> BaseTTSEngine:
    if args.engine == "mms":
        from .engines.mms_engine import MMSEngine

        return MMSEngine(model_dir=args.model_dir, device=args.device)

    if args.engine == "piper":
        from .engines.piper_engine import PiperEngine

        return PiperEngine(
            model_path=args.piper_model,
            config_path=args.piper_config,
            piper_bin=args.piper_bin,
            speaker=args.speaker,
            length_scale=args.length_scale,
            noise_scale=args.noise_scale,
            noise_w_scale=args.noise_w_scale,
        )

    if args.engine == "xtts":
        from .engines.xtts_engine import XTTSEngine

        return XTTSEngine(
            model_dir=args.xtts_model_dir,
            speaker_wav=args.xtts_speaker_wav,
            language=args.xtts_language,
            device=args.device,
            progress_bar=False,
        )

    raise ValueError(f"Unsupported engine: {args.engine}")


def log_run_configuration(
    args: argparse.Namespace, lines: list[str], segment_texts: list[str]
) -> None:
    print_step(f"Pause between spoken items: {args.pause:.2f}s")
    print_step(f"Question repeats: {args.question_repeats}")
    print_step(f"Answer repeats: {args.answer_repeats}")
    print_step(f"Speaking rate: {args.speaking_rate:.2f}")
    print_step(f"Input text lines: {len(lines)}")
    print_step(f"Speech segments to generate: {len(segment_texts)}")


def main() -> int:
    args = parse_args()

    try:
        started_at = time.perf_counter()

        validate_args(args)
        ensure_ffmpeg()

        print_step(f"Reading input file: {args.input}")
        input_text = read_input_text(args.input)
        if input_text == "":
            raise ValueError("Input file is empty.")

        lines = prepare_lines(input_text)
        if not lines:
            raise ValueError("Input file contains no non-empty text lines to synthesize.")

        segment_texts = build_segment_plan(
            lines,
            question_repeats=args.question_repeats,
            answer_repeats=args.answer_repeats,
        )
        log_run_configuration(args, lines, segment_texts)

        args.output.parent.mkdir(parents=True, exist_ok=True)

        engine = create_engine(args)
        print_step(f"Loading engine: {args.engine}")
        with closing(engine):
            engine.load()

            with tempfile.TemporaryDirectory(prefix="tts_build_") as tmp_dir_str:
                tmp_dir = Path(tmp_dir_str)
                parts = render_audio_parts(
                    engine=engine,
                    segment_texts=segment_texts,
                    tmp_dir=tmp_dir,
                    pause=args.pause,
                    speaking_rate=args.speaking_rate,
                    info_callback=print_step,
                    progress_callback=print_progress,
                )
                print_step("Merging audio parts into final MP3...")
                concat_to_mp3(parts, args.output, progress_callback=print_progress)

        elapsed = time.perf_counter() - started_at
        size_mb = args.output.stat().st_size / (1024 * 1024)

        print_step(f"Done: {args.output}")
        print_step(f"Output size: {size_mb:.2f} MB")
        print_step(f"Elapsed time: {elapsed:.2f}s")
        return 0
    except OSError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
