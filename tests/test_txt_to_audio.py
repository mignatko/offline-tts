from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pytest

import txt_to_audio


def make_args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    input_path = tmp_path / "input.txt"
    input_path.write_text("Kerdes?\nValasz.\n", encoding="utf-8")

    output_path = tmp_path / "output.mp3"
    model_dir = tmp_path / "mms-model"
    model_dir.mkdir()
    xtts_model_dir = tmp_path / "xtts-model"
    xtts_model_dir.mkdir()
    speaker_wav = tmp_path / "speaker.wav"
    speaker_wav.write_bytes(b"wav")

    values: dict[str, object] = {
        "engine": "mms",
        "input": input_path,
        "output": output_path,
        "pause": 1.0,
        "question_repeats": 1,
        "answer_repeats": 1,
        "speaking_rate": 1.0,
        "model_dir": model_dir,
        "device": "cpu",
        "piper_model": None,
        "piper_config": None,
        "piper_bin": "piper",
        "speaker": None,
        "length_scale": None,
        "noise_scale": None,
        "noise_w_scale": None,
        "xtts_model_dir": xtts_model_dir,
        "xtts_speaker_wav": speaker_wav,
        "xtts_language": "hu",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_prepare_lines_strips_blank_lines() -> None:
    assert txt_to_audio.prepare_lines(" Kerdes? \n\n Valasz. \r\n") == ["Kerdes?", "Valasz."]


def test_read_input_text_normalizes_newlines(tmp_path: Path) -> None:
    input_path = tmp_path / "sample.txt"
    input_path.write_text("a\r\nb\rc\n", encoding="utf-8")

    assert txt_to_audio.read_input_text(input_path) == "a\nb\nc\n"


def test_validate_args_rejects_missing_piper_model(tmp_path: Path) -> None:
    args = make_args(tmp_path, engine="piper", piper_model=None)

    with pytest.raises(ValueError, match="--piper-model is required"):
        txt_to_audio.validate_args(args)


def test_validate_args_rejects_missing_xtts_speaker(tmp_path: Path) -> None:
    args = make_args(tmp_path, engine="xtts", xtts_speaker_wav=None)

    with pytest.raises(ValueError, match="--xtts-speaker-wav is required"):
        txt_to_audio.validate_args(args)


def test_validate_args_accepts_existing_xtts_paths(tmp_path: Path) -> None:
    args = make_args(tmp_path, engine="xtts")

    txt_to_audio.validate_args(args)


def test_apply_tempo_filter_moves_file_when_rate_is_one(tmp_path: Path) -> None:
    input_wav = tmp_path / "input.wav"
    output_wav = tmp_path / "output.wav"
    input_wav.write_bytes(b"wav")

    txt_to_audio.apply_tempo_filter(input_wav, output_wav, 1.0)

    assert not input_wav.exists()
    assert output_wav.read_bytes() == b"wav"


@pytest.mark.parametrize(
    ("speaking_rate", "expected_chain"),
    [
        (4.0, "atempo=2.000000,atempo=2.000000"),
        (0.25, "atempo=0.500000,atempo=0.500000"),
        (1.5, "atempo=1.500000"),
    ],
)
def test_apply_tempo_filter_builds_expected_ffmpeg_chain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    speaking_rate: float,
    expected_chain: str,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(txt_to_audio.subprocess, "run", fake_run)

    txt_to_audio.apply_tempo_filter(tmp_path / "input.wav", tmp_path / "output.wav", speaking_rate)

    assert captured["cmd"][5] == expected_chain


def test_concat_to_mp3_invokes_ffmpeg_with_concat_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    part_a = tmp_path / "a.wav"
    part_b = tmp_path / "b.wav"
    output_path = tmp_path / "final.mp3"
    for path in (part_a, part_b):
        path.write_bytes(b"wav")

    captured: dict[str, object] = {}

    def fake_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        concat_list = Path(cmd[7])
        captured["concat_contents"] = concat_list.read_text(encoding="utf-8")
        output_path.write_bytes(b"mp3")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(txt_to_audio.subprocess, "run", fake_run)

    txt_to_audio.concat_to_mp3([part_a, part_b], output_path)

    assert captured["cmd"][-1] == str(output_path)
    assert f"file '{part_a.as_posix()}'" in captured["concat_contents"]
    assert f"file '{part_b.as_posix()}'" in captured["concat_contents"]


def test_main_generates_segments_and_writes_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = make_args(
        tmp_path,
        question_repeats=2,
        answer_repeats=1,
        output=tmp_path / "result.mp3",
    )

    synth_calls: list[str] = []
    applied_tempo: list[tuple[Path, Path, float]] = []
    merged_parts: list[Path] = []

    class FakeEngine:
        def load(self) -> None:
            return None

        def synthesize_to_wav(self, text: str, output_path: Path) -> int:
            synth_calls.append(text)
            output_path.write_bytes(b"wav")
            return 22050

    def fake_generate_silence(duration: float, sample_rate: int, output_path: Path) -> None:
        assert duration == 1.0
        assert sample_rate == 22050
        output_path.write_bytes(b"silence")

    def fake_apply_tempo(input_wav: Path, output_wav: Path, speaking_rate: float) -> None:
        applied_tempo.append((input_wav, output_wav, speaking_rate))
        output_wav.write_bytes(input_wav.read_bytes())
        input_wav.unlink()

    def fake_concat(parts: list[Path], output_path: Path) -> None:
        merged_parts.extend(parts)
        output_path.write_bytes(b"mp3")

    monkeypatch.setattr(txt_to_audio, "parse_args", lambda: args)
    monkeypatch.setattr(txt_to_audio, "ensure_ffmpeg", lambda: None)
    monkeypatch.setattr(txt_to_audio, "create_engine", lambda actual_args: FakeEngine())
    monkeypatch.setattr(txt_to_audio, "generate_silence_wav", fake_generate_silence)
    monkeypatch.setattr(txt_to_audio, "apply_tempo_filter", fake_apply_tempo)
    monkeypatch.setattr(txt_to_audio, "concat_to_mp3", fake_concat)

    assert txt_to_audio.main() == 0
    assert synth_calls == ["Kerdes?", "Kerdes?", "Kerdes?", "Valasz."]
    assert len(applied_tempo) == 3
    assert len(merged_parts) == 5
    assert args.output.read_bytes() == b"mp3"
