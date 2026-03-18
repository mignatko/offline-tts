from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from types import TracebackType


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

    def close(self) -> None:
        """
        Release optional engine resources.
        Engines that keep large models or other heavyweight handles can override this.
        """
        return None

    def __enter__(self) -> BaseTTSEngine:
        self.load()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
