from pathlib import Path
from huggingface_hub import snapshot_download

target_dir = Path("models/xtts-v2")

snapshot_download(
    repo_id="coqui/XTTS-v2",
    local_dir=target_dir,
    local_dir_use_symlinks=False,
)

print(f"Downloaded to: {target_dir.resolve()}")