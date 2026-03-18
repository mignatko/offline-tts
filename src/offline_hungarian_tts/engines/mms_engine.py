from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoTokenizer, VitsModel

from .base import BaseTTSEngine


class MMSEngine(BaseTTSEngine):
    def __init__(self, model_dir: Path, device: str = "auto") -> None:
        self.model_dir = Path(model_dir)
        self.device = self._resolve_device(device)
        self.tokenizer = None
        self.model = None

    def _resolve_device(self, device_arg: str) -> str:
        if device_arg == "cpu":
            return "cpu"
        if device_arg == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS was requested but is not available.")
            return "mps"
        return "mps" if torch.backends.mps.is_available() else "cpu"

    def load(self) -> None:
        if not self.model_dir.exists():
            raise FileNotFoundError(f"MMS model directory does not exist: {self.model_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )
        self.model = VitsModel.from_pretrained(
            self.model_dir,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()

    def synthesize_to_wav(self, text: str, output_path: Path) -> int:
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("MMS engine is not loaded.")

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            waveform = self.model(**inputs).waveform

        audio = waveform.cpu().float().numpy().squeeze()
        sample_rate = int(self.model.config.sampling_rate)
        sf.write(str(output_path), audio, sample_rate)
        return sample_rate
