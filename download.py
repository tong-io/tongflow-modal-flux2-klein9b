"""Modal download entry for flux2-klein9b.

Run:
  modal run download.py::download

Requires Modal secret `huggingface` (HF_TOKEN) for gated repos.
Self-contained: do not import other local modules.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import modal



_cfg: dict[str, Any] = {}
_flux_hf = _cfg.get("fluxHf") if isinstance(_cfg.get("fluxHf"), dict) else {}

HF_KLEIN_KV = str(
    _flux_hf.get("kleinKvRepoId") or "black-forest-labs/FLUX.2-klein-9B-kv",
)
HF_DEV = str(_flux_hf.get("devRepoId") or "black-forest-labs/FLUX.2-dev")
HF_QWEN = str(_flux_hf.get("qwenRepoId") or "Qwen/Qwen3-8B-FP8")

KLEIN_KV_DIR = f"/models/{HF_KLEIN_KV}"
DEV_DIR = f"/models/{HF_DEV}"
QWEN_DIR = f"/models/{HF_QWEN}"

volume_name = str(_cfg.get("volumeName") or "models")
volume = modal.Volume.from_name(volume_name, create_if_missing=True)
model_downloader = modal.App("model_downloader")


@model_downloader.function(
    image=modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub>=0.34.0,<1.0"),
    volumes={"/models": volume},
    timeout=7200,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
def _download() -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    token = os.environ.get("HF_TOKEN") or None
    if not token:
        print("Warning: HF_TOKEN is empty; gated repos (FLUX.2-dev, etc.) will return 403.")

    try:
        kv_marker = os.path.join(KLEIN_KV_DIR, "flux-2-klein-9b-kv.safetensors")
        if not (os.path.exists(kv_marker) and os.path.getsize(kv_marker) > 1000):
            print(f"Downloading {HF_KLEIN_KV} -> {KLEIN_KV_DIR} ...")
            os.makedirs(KLEIN_KV_DIR, exist_ok=True)
            snapshot_download(
                repo_id=HF_KLEIN_KV,
                local_dir=KLEIN_KV_DIR,
                token=token,
                local_dir_use_symlinks=False,
            )
            print(f"Done: {KLEIN_KV_DIR}")

        ae_marker = os.path.join(DEV_DIR, "ae.safetensors")
        if not (os.path.exists(ae_marker) and os.path.getsize(ae_marker) > 1000):
            print(f"Downloading ae.safetensors from {HF_DEV} (gated; license required) ...")
            os.makedirs(DEV_DIR, exist_ok=True)
            hf_hub_download(
                repo_id=HF_DEV,
                filename="ae.safetensors",
                local_dir=DEV_DIR,
                token=token,
                local_dir_use_symlinks=False,
            )
            print(f"Done: {ae_marker}")

        qwen_marker = os.path.join(QWEN_DIR, "config.json")
        if not os.path.exists(qwen_marker):
            print(f"Downloading {HF_QWEN} -> {QWEN_DIR} ...")
            os.makedirs(QWEN_DIR, exist_ok=True)
            snapshot_download(
                repo_id=HF_QWEN,
                local_dir=QWEN_DIR,
                token=token,
                local_dir_use_symlinks=False,
            )
            print(f"Done: {QWEN_DIR}")

        volume.commit()
    except Exception as e:
        raise RuntimeError(
            "Hugging Face download failed. "
            "For black-forest-labs/FLUX.2-dev you must accept the model license on Hugging Face "
            "and use an HF_TOKEN that has access (Modal secret `huggingface`). "
            f"Original: {type(e).__name__}: {e}"
        ) from None


@model_downloader.local_entrypoint()
def download() -> None:
    _download.remote()
