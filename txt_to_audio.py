#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from offline_hungarian_tts.audio import apply_tempo_filter, concat_to_mp3, ensure_ffmpeg, read_input_text
from offline_hungarian_tts.cli import (
    create_engine,
    log_run_configuration,
    main,
    parse_args,
    print_progress,
    print_step,
    validate_args,
)
from offline_hungarian_tts.pipeline import build_segment_plan, prepare_lines, render_audio_parts


if __name__ == "__main__":
    raise SystemExit(main())
