from __future__ import annotations

from pathlib import Path

import soundfile as sf
from TTS.api import TTS
import torch

from .base import BaseTTSEngine


class XTTSEngine(BaseTTSEngine):
    def __init__(
        self,
        model_dir: str | Path,
        speaker_wav: str | Path,
        language: str = "hu",
        device: str = "cpu",
        progress_bar: bool = False,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.speaker_wav = Path(speaker_wav)
        self.language = language
        self.device = device
        self.progress_bar = progress_bar
        self.tts = None

    def _resolve_device(self) -> str:
        if self.device == "cpu":
            return "cpu"
        if self.device == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS was requested but is not available.")
            return "mps"
        return "cpu"

    def load(self) -> None:
        if not self.model_dir.exists():
            raise FileNotFoundError(f"XTTS model directory does not exist: {self.model_dir}")

        if not self.speaker_wav.exists():
            raise FileNotFoundError(
                f"XTTS speaker reference file does not exist: {self.speaker_wav}"
            )

        config_path = self.model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"XTTS config.json was not found: {config_path}")

        self.tts = TTS(
            model_path=str(self.model_dir),
            config_path=str(config_path),
            progress_bar=self.progress_bar,
            gpu=False,
        ).to(self._resolve_device())

    def synthesize_to_wav(self, text: str, output_path: Path) -> int:
        if self.tts is None:
            raise RuntimeError("XTTS engine is not loaded.")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.tts.tts_to_file(
            text=text,
            file_path=str(output_path),
            speaker_wav=str(self.speaker_wav),
            language=self.language,
        )

        return int(sf.info(str(output_path)).samplerate)

    def close(self) -> None:
        self.tts = None
