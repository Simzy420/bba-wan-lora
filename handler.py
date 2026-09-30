"""Runpod serverless handler. Pod boots use bootstrap.py via entrypoint.py."""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def write_tree(work: Path, payload: dict) -> None:
    work.mkdir(parents=True, exist_ok=True)
    (work / "output").mkdir(exist_ok=True)
    config = payload.get("config") or {}
    (work / "config.json").write_text(json.dumps(config), encoding="utf-8")
    for key, filename in (
        ("dataset_toml", "dataset.toml"),
        ("train_commands_sh", "train_commands.sh"),
        ("train_script_py", "train_wan22_lora.py"),
        ("readme", "README.txt"),
    ):
        text = payload.get(key)
        if isinstance(text, str) and text.strip():
            (work / filename).write_text(text, encoding="utf-8")
    images = work / "images"
    images.mkdir(exist_ok=True)
    for image in payload.get("images") or []:
        name = str(image.get("name") or "image.jpg")
        if "/" in name or "\\" in name or name.startswith("."):
            continue
        raw = image.get("data_base64") or ""
        if isinstance(raw, str) and raw:
            (images / name).write_bytes(base64.b64decode(raw))
        caption = image.get("caption")
        if isinstance(caption, str):
            (images / Path(name).with_suffix(".txt").name).write_text(caption, encoding="utf-8")


def handler(event: dict) -> dict:
    payload = event.get("input") or {}
    work = Path(tempfile.mkdtemp(prefix="bba-wan-"))
    write_tree(work, payload)
    script = work / "train_wan22_lora.py"
    if not script.is_file():
        return {"ok": False, "error": "train_wan22_lora.py was not in the job.", "mode": "scaffold"}
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=work,
        check=False,
        capture_output=True,
        text=True,
    )
    status_path = work / "output" / "status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
    else:
        status = {
            "ok": False,
            "error": completed.stderr[-500:] or "Trainer did not write status.json.",
            "mode": "scaffold",
            "logs": [completed.stdout[-500:]],
        }
    bundle = work / "output" / "bba_wan22_lora.zip"
    if bundle.is_file() and bundle.stat().st_size <= 4_500_000:
        status["bundle_base64"] = base64.b64encode(bundle.read_bytes()).decode("ascii")
        status["filename"] = bundle.name
    elif bundle.is_file():
        status["bundle_too_large"] = True
        status["bytes"] = bundle.stat().st_size
    status["exit_code"] = completed.returncode
    return status


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
