from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the local XTTS-v2 snapshot into the models folder."
    )
    parser.add_argument(
        "--repo-id",
        default="coqui/XTTS-v2",
        help="Hugging Face repository id to download.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("models/xtts-v2"),
        help="Local directory where XTTS-v2 should be saved.",
    )
    return parser.parse_args()


def download_xtts_v2(repo_id: str, target_dir: Path) -> Path:
    snapshot_download(
        repo_id=repo_id,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded to: {target_dir.resolve()}")
    return target_dir


def main() -> int:
    args = parse_args()
    download_xtts_v2(args.repo_id, args.target_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
