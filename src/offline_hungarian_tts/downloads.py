from __future__ import annotations

from pathlib import Path


def download_mms_hu(model_id: str, target_dir: Path) -> Path:
    from transformers import AutoTokenizer, VitsModel

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


def download_xtts_v2(repo_id: str, target_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        local_dir=target_dir,
    )
    print(f"Downloaded to: {target_dir.resolve()}")
    return target_dir
