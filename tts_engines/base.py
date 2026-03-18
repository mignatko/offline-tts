from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseTTSEngine(ABC):
    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def synthesize_to_wav(self, text: str, output_path: Path) -> int:
        """
        Synthesize text to a WAV file.
        Returns the sample rate.
        """
        pass