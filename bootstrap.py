"""Download the phone's dataset zip and run the trainer. Used by GPU pods."""

from __future__ import annotations

import os
import sys
import urllib.request
import zipfile
from pathlib import Path


def main() -> None:
    url = os.environ.get("DATASET_URL", "").strip()
    if not url:
        raise SystemExit("DATASET_URL is required for pod mode.")
    root = Path("/workspace/bba-job")
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "dataset.zip"
    urllib.request.urlretrieve(url, archive)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(root)
    script = root / "train_wan22_lora.py"
    if not script.is_file():
        raise SystemExit("Dataset zip did not include train_wan22_lora.py.")
    os.chdir(root)
    os.execv(sys.executable, [sys.executable, str(script), "--serve", "8080"])


if __name__ == "__main__":
    main()
