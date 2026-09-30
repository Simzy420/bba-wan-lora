"""Pod image runs the dataset bootstrap. Serverless starts handler.py."""

from __future__ import annotations

import os


def main() -> None:
    if os.environ.get("DATASET_URL", "").strip():
        from bootstrap import main as boot

        boot()
        return
    import runpod

    from handler import handler

    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
