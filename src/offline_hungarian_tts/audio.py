from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from shutil import which


def ensure_ffmpeg() -> None:
    if which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found in PATH.")


def read_input_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {path}")

    text = path.read_text(encoding="utf-8")
    return text.replace("\r\n", "\n").replace("\r", "\n")


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


def concat_to_mp3(parts: list[Path], output_path: Path, progress_callback) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".txt", delete=False) as f:
        concat_list = Path(f.name)
        for part in parts:
            f.write(f"file '{part.as_posix()}'\n")

    try:
        progress_callback(1, 1, prefix="Merging")
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
