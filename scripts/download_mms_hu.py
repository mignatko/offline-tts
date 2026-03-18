from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer, VitsModel


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


def download_mms_hu(model_id: str, target_dir: Path) -> Path:
    print(f"Downloading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Downloading model: {model_id}")
    model = VitsModel.from_pretrained(model_id)

    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving tokenizer to: {target_dir}")
    tokenizer.save_pretrained(target_dir)

    print(f"Saving model to: {target_dir}")
    model.save_pretrained(target_dir)

    print("Done.")
    return target_dir


def main() -> int:
    args = parse_args()
    download_mms_hu(args.model_id, args.target_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
