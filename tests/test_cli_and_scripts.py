from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import offline_hungarian_tts.cli as txt_to_audio


def load_module(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_txt_to_audio_parse_args_parses_required_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "output.mp3"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "txt_to_audio.py",
            "--engine",
            "mms",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--question-repeats",
            "2",
            "--answer-repeats",
            "3",
        ],
    )

    args = txt_to_audio.parse_args()

    assert args.engine == "mms"
    assert args.input == input_path
    assert args.output == output_path
    assert args.question_repeats == 2
    assert args.answer_repeats == 3


def test_download_mms_hu_main_uses_reusable_function(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda *_args, **_kwargs: None
    )
    fake_transformers.VitsModel = SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda *_args, **_kwargs: None
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    module = load_module(
        "download_mms_hu_module",
        Path.cwd() / "scripts" / "download_mms_hu.py",
    )

    target_dir = tmp_path / "mms-target"
    called: dict[str, object] = {}

    def fake_download(model_id: str, actual_target_dir: Path) -> Path:
        called["model_id"] = model_id
        called["target_dir"] = actual_target_dir
        return actual_target_dir

    monkeypatch.setattr(module, "download_mms_hu", fake_download)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_mms_hu.py",
            "--model-id",
            "facebook/mms-tts-hun",
            "--target-dir",
            str(target_dir),
        ],
    )

    assert module.main() == 0
    assert called == {
        "model_id": "facebook/mms-tts-hun",
        "target_dir": target_dir,
    }


def test_download_xtts_v2_main_uses_reusable_function(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_hf_hub = ModuleType("huggingface_hub")
    fake_hf_hub.snapshot_download = lambda **_kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_hub)

    module = load_module(
        "download_xtts_v2_module",
        Path.cwd() / "scripts" / "download_xtts_v2.py",
    )

    target_dir = tmp_path / "xtts-target"
    called: dict[str, object] = {}

    def fake_download(repo_id: str, actual_target_dir: Path) -> Path:
        called["repo_id"] = repo_id
        called["target_dir"] = actual_target_dir
        return actual_target_dir

    monkeypatch.setattr(module, "download_xtts_v2", fake_download)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_xtts_v2.py",
            "--repo-id",
            "coqui/XTTS-v2",
            "--target-dir",
            str(target_dir),
        ],
    )

    assert module.main() == 0
    assert called == {
        "repo_id": "coqui/XTTS-v2",
        "target_dir": target_dir,
    }


def test_download_mms_hu_downloads_and_saves_to_target_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_transformers = ModuleType("transformers")
    saved: list[tuple[str, Path]] = []

    class FakeTokenizer:
        def save_pretrained(self, target_dir: Path) -> None:
            saved.append(("tokenizer", target_dir))

    class FakeModel:
        def save_pretrained(self, target_dir: Path) -> None:
            saved.append(("model", target_dir))

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id: str) -> FakeTokenizer:
            assert model_id == "facebook/mms-tts-hun"
            return FakeTokenizer()

    class FakeVitsModel:
        @staticmethod
        def from_pretrained(model_id: str) -> FakeModel:
            assert model_id == "facebook/mms-tts-hun"
            return FakeModel()

    fake_transformers.AutoTokenizer = FakeAutoTokenizer  # type: ignore[attr-defined]
    fake_transformers.VitsModel = FakeVitsModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    sys.modules.pop("offline_hungarian_tts.downloads", None)
    module = importlib.import_module("offline_hungarian_tts.downloads")
    target_dir = tmp_path / "downloaded-model"

    returned_dir = module.download_mms_hu("facebook/mms-tts-hun", target_dir)

    assert returned_dir == target_dir
    assert target_dir.exists()
    assert saved == [("tokenizer", target_dir), ("model", target_dir)]


def test_download_xtts_v2_downloads_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_hf_hub = ModuleType("huggingface_hub")
    captured: dict[str, object] = {}

    def fake_snapshot_download(**kwargs: object) -> None:
        captured.update(kwargs)

    fake_hf_hub.snapshot_download = fake_snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_hub)
    sys.modules.pop("offline_hungarian_tts.downloads", None)
    module = importlib.import_module("offline_hungarian_tts.downloads")
    target_dir = tmp_path / "xtts-download"

    returned_dir = module.download_xtts_v2("coqui/XTTS-v2", target_dir)

    assert returned_dir == target_dir
    assert captured == {
        "repo_id": "coqui/XTTS-v2",
        "local_dir": target_dir,
    }
