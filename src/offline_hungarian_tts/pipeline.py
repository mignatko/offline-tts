from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .audio import apply_tempo_filter, generate_silence_wav
from .engines.base import BaseTTSEngine


def prepare_lines(input_text: str) -> list[str]:
    return [line.strip() for line in input_text.split("\n") if line.strip()]


def build_segment_plan(lines: list[str], question_repeats: int, answer_repeats: int) -> list[str]:
    segment_texts: list[str] = []
    for idx, text in enumerate(lines, start=1):
        repeats = question_repeats if idx % 2 == 1 else answer_repeats
        segment_texts.extend([text] * repeats)
    return segment_texts


def render_audio_parts(
    engine: BaseTTSEngine,
    segment_texts: list[str],
    tmp_dir: Path,
    pause: float,
    speaking_rate: float,
    info_callback: Callable[[str], None],
    progress_callback: Callable[..., None],
) -> list[Path]:
    info_callback("Generating first speech sample to determine sampling rate...")
    probe_wav = tmp_dir / "probe.wav"
    sample_rate = engine.synthesize_to_wav(segment_texts[0], probe_wav)
    probe_wav.unlink(missing_ok=True)

    silence_wav = tmp_dir / "silence.wav"
    info_callback(f"Generating silence segment ({pause:.2f}s)...")
    generate_silence_wav(pause, sample_rate, silence_wav)

    parts: list[Path] = []

    info_callback("Generating speech audio...")

    for idx, text in enumerate(segment_texts, start=1):
        raw_wav = tmp_dir / f"speech_raw_{idx:05d}.wav"
        final_wav = tmp_dir / f"speech_{idx:05d}.wav"

        engine.synthesize_to_wav(text, raw_wav)
        apply_tempo_filter(raw_wav, final_wav, speaking_rate)

        if raw_wav.exists():
            raw_wav.unlink()

        parts.append(final_wav)

        if idx != len(segment_texts):
            parts.append(silence_wav)

        progress_callback(idx, len(segment_texts), prefix="Synthesizing")

    return parts
