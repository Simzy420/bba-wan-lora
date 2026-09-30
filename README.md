# Wan 2.2 LoRA worker

Phone jobs do not open Jupyter. This image is the thing `RUNPOD_TRAIN_IMAGE` should point at.

- **Serverless:** leave `DATASET_URL` unset. Runpod starts `handler.py`. Set the endpoint id as `RUNPOD_TRAIN_ENDPOINT_ID`. Keep `workersMin` at 0.
- **Pod:** Wan Desk sets `DATASET_URL`. `bootstrap.py` downloads the zip and serves status on port 8080. The app stops and, by default, terminates the pod when training finishes or fails. `terminateAfter` is also set to 90 minutes as a backstop.

Real musubi-tuner training runs only when the container has:

- `MUSUBI_DIR` — checkout with `src/musubi_tuner/wan_train_network.py`
- `WAN_DIT_LOW` / `WAN_DIT_HIGH` — fp16 Wan 2.2 DiTs (not fp8_scaled)
- `WAN_VAE` — Wan 2.1 VAE
- `WAN_T5` — UMT5-XXL encoder

Without those files the same script still finishes: it writes a safetensors metadata shell and `train_commands.sh`, then the phone stops the GPU. That shell is not a trained LoRA.

24GB defaults live in `train_commands.sh`: `adamw8bit`, `--fp8_base`, `--blocks_to_swap 18`, low-noise timesteps 0–875 then high-noise 875–1000. Do not point this image at an RTX PRO 6000. Wan Desk refuses that GPU.
