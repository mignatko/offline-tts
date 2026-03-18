from typing import Any

__all__ = ["MMSEngine", "PiperEngine", "XTTSEngine"]


def __getattr__(name: str) -> Any:
    if name == "MMSEngine":
        from .mms_engine import MMSEngine

        return MMSEngine
    if name == "PiperEngine":
        from .piper_engine import PiperEngine

        return PiperEngine
    if name == "XTTSEngine":
        from .xtts_engine import XTTSEngine

        return XTTSEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
