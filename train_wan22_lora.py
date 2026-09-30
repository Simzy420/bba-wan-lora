"""Wan 2.2 LoRA job runner for BBA Wan Desk.

Scaffold mode writes a real safetensors header plus the musubi command.
Real training runs only when MUSUBI_DIR and the Wan 2.2 weight files exist.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import threading
import time
import zipfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def write_safetensors(path: Path, metadata: dict[str, str]) -> None:
    header = json.dumps({"__metadata__": metadata}, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header)))
        handle.write(header)


def write_status(output: Path, payload: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "status.json").write_text(json.dumps(payload), encoding="utf-8")


def targets(dit: str) -> list[str]:
    if dit == "high":
        return ["high"]
    if dit == "low":
        return ["low"]
    return ["low", "high"]


def model_env(which: str) -> str:
    return "WAN_DIT_HIGH" if which == "high" else "WAN_DIT_LOW"


def weights_ready(job: Path, dit: str) -> tuple[bool, str]:
    musubi = os.environ.get("MUSUBI_DIR", "").strip()
    if not musubi or not Path(musubi, "src/musubi_tuner/wan_train_network.py").is_file():
        return False, "MUSUBI_DIR is not a musubi-tuner checkout."
    for key in ("WAN_VAE", "WAN_T5"):
        value = os.environ.get(key, "").strip()
        if not value or not Path(value).is_file():
            return False, f"{key} is not a file on this machine."
    for which in targets(dit):
        key = model_env(which)
        value = os.environ.get(key, "").strip()
        if not value or not Path(value).is_file():
            return False, f"{key} is not a file on this machine."
    if not (job / "train_commands.sh").is_file():
        return False, "train_commands.sh is missing from the dataset zip."
    return True, ""


def finalize_toml(job: Path) -> None:
    path = job / "dataset.toml"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8").replace("__JOB__", str(job))
    path.write_text(text, encoding="utf-8")


def serve_output(output: Path, port: int) -> None:
    directory = str(output)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=directory, **kwargs)

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()


def pack_bundle(job: Path, output: Path) -> None:
    bundle = output / "bba_wan22_lora.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in output.glob("*.safetensors"):
            archive.write(path, f"output/{path.name}")
        for name in ("README.txt", "train_commands.sh", "dataset.toml", "config.json"):
            source = job / name
            if source.is_file():
                archive.write(source, name)
        log = output / "train.log"
        if log.is_file():
            archive.write(log, "train.log")


def run_training(job: Path) -> dict[str, Any]:
    output = job / "output"
    output.mkdir(parents=True, exist_ok=True)
    config = {}
    config_path = job / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    trigger = str(config.get("trigger") or "bba")
    dit = str(config.get("dit") or "both")
    rank = str(config.get("rank") or "16")
    steps = str(config.get("steps") or "400")
    logs: list[str] = ["Trainer started."]
    write_status(
        output,
        {
            "phase": "preparing",
            "progress": 8,
            "label": "Preparing the dataset",
            "logs": logs,
            "done": False,
            "ok": True,
            "mode": "scaffold",
            "artifact": "bba_wan22_lora.zip",
        },
    )
    finalize_toml(job)
    ready, reason = weights_ready(job, dit)
    mode = "scaffold"
    note = (
        "Scaffold only. The safetensors files are metadata shells. "
        "Mount musubi-tuner and the Wan 2.2 fp16 DiTs, then rerun train_commands.sh."
    )
    if ready:
        logs.append("Weights found. Starting musubi-tuner.")
        write_status(
            output,
            {
                "phase": "training",
                "progress": 20,
                "label": "Training Wan 2.2 LoRA",
                "logs": logs,
                "done": False,
                "ok": True,
                "mode": "trained",
                "artifact": "bba_wan22_lora.zip",
            },
        )
        log_path = output / "train.log"
        with log_path.open("w", encoding="utf-8") as handle:
            proc = subprocess.run(
                ["bash", str(job / "train_commands.sh")],
                cwd=job,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
        logs.extend(tail)
        if proc.returncode != 0:
            note = f"musubi-tuner exited {proc.returncode}."
            write_status(
                output,
                {
                    "phase": "failed",
                    "progress": 100,
                    "label": "Training failed",
                    "logs": logs,
                    "done": True,
                    "ok": False,
                    "error": note,
                    "mode": "scaffold",
                    "artifact": "bba_wan22_lora.zip",
                },
            )
            return {"ok": False, "mode": "scaffold", "note": note, "logs": logs}
        mode = "trained"
        note = "musubi-tuner finished. Download the safetensors from this bundle."
    else:
        logs.append(reason)
        logs.append("Writing scaffold safetensors and leaving the GPU job finish path.")
        for which in targets(dit):
            filename = output / f"{trigger}_wan22_{which}.safetensors"
            write_safetensors(
                filename,
                {
                    "format": "bba-wan-desk",
                    "trained": "false",
                    "mode": "scaffold",
                    "trigger": trigger,
                    "dit": which,
                    "rank": rank,
                    "steps": steps,
                    "note": "Metadata shell. Not trained Wan weights.",
                },
            )
    mirror = os.environ.get("WAN_OUTPUT_DIR", "").strip()
    if mirror:
        dest = Path(mirror)
        dest.mkdir(parents=True, exist_ok=True)
        for path in output.glob("*.safetensors"):
            shutil.copy2(path, dest / path.name)
        logs.append(f"Copied weights to {dest}.")
    pack_bundle(job, output)
    logs.append("Bundle written. Stop the GPU from Wan Desk if it is still up.")
    payload = {
        "phase": "done",
        "progress": 100,
        "label": "LoRA ready" if mode == "trained" else "Scaffold bundle ready",
        "logs": logs[-40:],
        "done": True,
        "ok": True,
        "mode": "trained" if mode == "trained" else "scaffold",
        "note": note,
        "artifact": "bba_wan22_lora.zip",
    }
    write_status(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", type=int, default=0)
    args = parser.parse_args()
    job = Path.cwd()
    output = job / "output"
    output.mkdir(parents=True, exist_ok=True)
    if args.serve:
        write_status(
            output,
            {
                "phase": "starting",
                "progress": 5,
                "label": "Trainer is online",
                "logs": ["HTTP status is up. Training starts next."],
                "done": False,
                "ok": True,
                "mode": "scaffold",
                "artifact": "bba_wan22_lora.zip",
            },
        )
        serve_output(output, args.serve)
    try:
        run_training(job)
    except Exception as exc:  # noqa: BLE001 - surface any trainer failure to the phone
        write_status(
            output,
            {
                "phase": "failed",
                "progress": 100,
                "label": "Training failed",
                "logs": [str(exc)],
                "done": True,
                "ok": False,
                "error": str(exc),
                "mode": "scaffold",
                "artifact": "bba_wan22_lora.zip",
            },
        )
    if args.serve:
        while True:
            time.sleep(30)


if __name__ == "__main__":
    main()
