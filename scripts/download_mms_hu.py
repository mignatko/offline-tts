from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from offline_hungarian_tts.downloads import download_mms_hu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the local MMS Hungarian model into the models folder."
    )
    parser.add_argument(
        "--model-id",
        default="facebook/mms-tts-hun",
        help="Hugging Face model id to download.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("models/mms-tts-hun"),
        help="Local directory where the MMS model should be saved.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    download_mms_hu(args.model_id, args.target_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
