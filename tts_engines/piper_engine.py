from __future__ import annotations

import subprocess
import wave
from pathlib import Path

from .base import BaseTTSEngine


class PiperEngine(BaseTTSEngine):
    def __init__(
        self,
        model_path: Path,
        config_path: Path | None = None,
        piper_bin: str = "piper",
        speaker: int | None = None,
        length_scale: float | None = None,
        noise_scale: float | None = None,
        noise_w_scale: float | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.config_path = Path(config_path) if config_path else Path(str(model_path) + ".json")
        self.piper_bin = piper_bin
        self.speaker = speaker
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w_scale = noise_w_scale

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Piper model file does not exist: {self.model_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Piper config file does not exist: {self.config_path}")

    def synthesize_to_wav(self, text: str, output_path: Path) -> int:
        cmd = [
            self.piper_bin,
            "--model",
            str(self.model_path),
            "--config",
            str(self.config_path),
            "--output_file",
            str(output_path),
        ]

        if self.speaker is not None:
            cmd += ["--speaker", str(self.speaker)]

        if self.length_scale is not None:
            cmd += ["--length_scale", str(self.length_scale)]

        if self.noise_scale is not None:
            cmd += ["--noise_scale", str(self.noise_scale)]

        if self.noise_w_scale is not None:
            cmd += ["--noise_w", str(self.noise_w_scale)]

        try:
            subprocess.run(
                cmd,
                input=text,
                text=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            details = f": {stderr}" if stderr else "."
            raise RuntimeError(f"Piper synthesis failed{details}") from exc

        with wave.open(str(output_path), "rb") as wav_file:
            return wav_file.getframerate()
