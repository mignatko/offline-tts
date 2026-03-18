from __future__ import annotations

import importlib
import sys
import wave
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tts_engines.piper_engine import PiperEngine


def test_piper_engine_load_rejects_missing_model(tmp_path: Path) -> None:
    engine = PiperEngine(model_path=tmp_path / "missing.onnx")

    with pytest.raises(FileNotFoundError, match="Piper model file does not exist"):
        engine.load()


def test_piper_engine_synthesize_to_wav_invokes_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "voice.onnx"
    config_path = tmp_path / "voice.onnx.json"
    output_path = tmp_path / "out.wav"
    model_path.write_bytes(b"model")
    config_path.write_text("{}", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_run(cmd: list[str], **kwargs: object) -> None:
        captured["cmd"] = cmd
        captured["input"] = kwargs["input"]
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            wav_file.writeframes(b"\x00\x00")

    monkeypatch.setattr("tts_engines.piper_engine.subprocess.run", fake_run)

    engine = PiperEngine(
        model_path=model_path,
        config_path=config_path,
        piper_bin="piper",
        speaker=2,
        length_scale=1.1,
        noise_scale=0.4,
        noise_w_scale=0.8,
    )

    sample_rate = engine.synthesize_to_wav("Szia", output_path)

    assert sample_rate == 22050
    assert captured["input"] == "Szia"
    assert captured["cmd"] == [
        "piper",
        "--model",
        str(model_path),
        "--config",
        str(config_path),
        "--output_file",
        str(output_path),
        "--speaker",
        "2",
        "--length_scale",
        "1.1",
        "--noise_scale",
        "0.4",
        "--noise_w",
        "0.8",
    ]


def test_piper_engine_raises_runtime_error_with_stderr_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "voice.onnx"
    config_path = tmp_path / "voice.onnx.json"
    output_path = tmp_path / "out.wav"
    model_path.write_bytes(b"model")
    config_path.write_text("{}", encoding="utf-8")

    def fake_run(cmd: list[str], **kwargs: object) -> None:
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=cmd,
            stderr="missing phonemes",
        )

    import subprocess

    monkeypatch.setattr("tts_engines.piper_engine.subprocess.run", fake_run)

    engine = PiperEngine(
        model_path=model_path,
        config_path=config_path,
        piper_bin="piper",
    )

    with pytest.raises(RuntimeError, match="missing phonemes"):
        engine.synthesize_to_wav("Szia", output_path)


def test_mms_engine_load_and_synthesize_with_fake_dependencies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_soundfile = ModuleType("soundfile")
    fake_torch = ModuleType("torch")
    fake_transformers = ModuleType("transformers")
    writes: list[tuple[str, object, int]] = []

    class FakeArray:
        def squeeze(self) -> str:
            return "audio-array"

    class FakeWaveform:
        def cpu(self) -> "FakeWaveform":
            return self

        def float(self) -> "FakeWaveform":
            return self

        def numpy(self) -> FakeArray:
            return FakeArray()

    class FakeInputs(dict[str, object]):
        def to(self, device: str) -> "FakeInputs":
            self["device"] = device
            return self

    class FakeTokenizer:
        def __call__(self, text: str, return_tensors: str) -> FakeInputs:
            assert text == "Szia"
            assert return_tensors == "pt"
            return FakeInputs(text=text)

    class FakeModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(sampling_rate=16000)
            self.loaded_to: str | None = None
            self.eval_called = False

        def to(self, device: str) -> "FakeModel":
            self.loaded_to = device
            return self

        def eval(self) -> None:
            self.eval_called = True

        def __call__(self, **_: object) -> SimpleNamespace:
            return SimpleNamespace(waveform=FakeWaveform())

    fake_model = FakeModel()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_dir: Path, local_files_only: bool) -> FakeTokenizer:
            assert local_files_only is True
            assert model_dir == tmp_path
            return FakeTokenizer()

    class FakeVitsModel:
        @staticmethod
        def from_pretrained(model_dir: Path, local_files_only: bool) -> FakeModel:
            assert local_files_only is True
            assert model_dir == tmp_path
            return fake_model

    class FakeNoGrad:
        def __enter__(self) -> None:
            return None

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    fake_soundfile.write = lambda path, audio, sample_rate: writes.append(
        (path, audio, sample_rate)
    )
    fake_torch.no_grad = lambda: FakeNoGrad()
    fake_torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    fake_transformers.VitsModel = FakeVitsModel

    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    sys.modules.pop("tts_engines.mms_engine", None)

    module = importlib.import_module("tts_engines.mms_engine")
    engine = module.MMSEngine(model_dir=tmp_path, device="cpu")
    engine.load()

    sample_rate = engine.synthesize_to_wav("Szia", tmp_path / "out.wav")

    assert fake_model.loaded_to == "cpu"
    assert fake_model.eval_called is True
    assert sample_rate == 16000
    assert writes == [(str(tmp_path / "out.wav"), "audio-array", 16000)]


def test_xtts_engine_load_and_synthesize_with_fake_dependencies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_tts_package = ModuleType("TTS")
    fake_tts_api = ModuleType("TTS.api")
    fake_soundfile = ModuleType("soundfile")
    fake_torch = ModuleType("torch")
    captured: dict[str, object] = {}

    class FakeTTS:
        def __init__(
            self, model_path: str, config_path: str, progress_bar: bool, gpu: bool
        ) -> None:
            captured["init"] = {
                "model_path": model_path,
                "config_path": config_path,
                "progress_bar": progress_bar,
                "gpu": gpu,
            }

        def to(self, device: str) -> "FakeTTS":
            captured["device"] = device
            return self

        def tts_to_file(
            self, text: str, file_path: str, speaker_wav: str, language: str
        ) -> None:
            captured["tts_to_file"] = {
                "text": text,
                "file_path": file_path,
                "speaker_wav": speaker_wav,
                "language": language,
            }

    fake_tts_api.TTS = FakeTTS
    fake_tts_package.api = fake_tts_api
    fake_soundfile.info = lambda path: SimpleNamespace(samplerate=44100)
    fake_torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "TTS", fake_tts_package)
    monkeypatch.setitem(sys.modules, "TTS.api", fake_tts_api)
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    sys.modules.pop("tts_engines.xtts_engine", None)

    model_dir = tmp_path / "xtts"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    speaker_wav = tmp_path / "speaker.wav"
    speaker_wav.write_bytes(b"wav")

    module = importlib.import_module("tts_engines.xtts_engine")
    engine = module.XTTSEngine(
        model_dir=model_dir,
        speaker_wav=speaker_wav,
        language="hu",
        progress_bar=False,
    )
    engine.load()

    sample_rate = engine.synthesize_to_wav("Szia", tmp_path / "out.wav")

    assert captured["device"] == "cpu"
    assert captured["init"] == {
        "model_path": str(model_dir),
        "config_path": str(model_dir / "config.json"),
        "progress_bar": False,
        "gpu": False,
    }
    assert captured["tts_to_file"] == {
        "text": "Szia",
        "file_path": str(tmp_path / "out.wav"),
        "speaker_wav": str(speaker_wav),
        "language": "hu",
    }
    assert sample_rate == 44100


def test_xtts_engine_respects_explicit_mps_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_tts_package = ModuleType("TTS")
    fake_tts_api = ModuleType("TTS.api")
    fake_soundfile = ModuleType("soundfile")
    fake_torch = ModuleType("torch")
    captured: dict[str, object] = {}

    class FakeTTS:
        def __init__(self, **_: object) -> None:
            return None

        def to(self, device: str) -> "FakeTTS":
            captured["device"] = device
            return self

    fake_tts_api.TTS = FakeTTS
    fake_tts_package.api = fake_tts_api
    fake_soundfile.info = lambda path: SimpleNamespace(samplerate=24000)
    fake_torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "TTS", fake_tts_package)
    monkeypatch.setitem(sys.modules, "TTS.api", fake_tts_api)
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    sys.modules.pop("tts_engines.xtts_engine", None)

    model_dir = tmp_path / "xtts"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    speaker_wav = tmp_path / "speaker.wav"
    speaker_wav.write_bytes(b"wav")

    module = importlib.import_module("tts_engines.xtts_engine")
    engine = module.XTTSEngine(
        model_dir=model_dir,
        speaker_wav=speaker_wav,
        language="hu",
        device="mps",
    )
    engine.load()

    assert captured["device"] == "mps"
