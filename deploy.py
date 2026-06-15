"""
FLUX.2-klein-9B **KV**: image edit / multi-reference edit / text-to-image (reference images flow through KV cache; cheaper for multi-ref).

Aligned with other scripts in this directory: shared ``models`` volume, dedicated ``model_downloader``, ``git clone`` upstream inference repo inside the image.

Weights: ``black-forest-labs/FLUX.2-klein-9B-kv``, filename ``flux-2-klein-9b-kv.safetensors``; env var ``KLEIN_9B_KV_MODEL_PATH``.

Why also touch ``FLUX.2-dev`` in addition to Klein?
  **The Klein flow-model weights do not include the image VAE.**
  The edit pipeline needs: source image → **VAE encode** → latent conditioning → **VAE decode** → output image.
  The official ``flux2`` code (``util.py``) hard-codes the VAE to the single file ``ae.safetensors``, **hosted at**
  `black-forest-labs/FLUX.2-dev <https://huggingface.co/black-forest-labs/FLUX.2-dev>`_,
  so the downloader pulls **only that file**, **not** the full 32B ``FLUX.2 [dev]`` checkpoint.

  The Klein repo also contains a diffusers-style ``vae/``, but it is incompatible with the **native ``AutoEncoder`` + ``ae.safetensors``**
  loader used by this script — it cannot be substituted without modifying ``flux2`` source.

  If you already have a compatible ``ae.safetensors`` elsewhere, set env var ``AE_MODEL_PATH`` to that file and skip the
  ``FLUX.2-dev`` download (you are responsible for ensuring it matches the FLUX.2 training run).

The ``FLUX.2-dev`` repo is gated: after accepting the terms on Hugging Face, write the **same account's** ``HF_TOKEN``
into the Modal Secret ``huggingface`` (same as ``ltx.py``); otherwise fetching ``ae.safetensors`` returns 403.

Deploy::

    modal deploy gpu/flux2_klein9b.py

Pre-fetch weights::

    modal run gpu/flux2_klein9b.py::download
"""

from __future__ import annotations

import gc
import io
import json
import os
import random
from pathlib import Path
from typing import Any, List, Optional, cast

import modal
from tongflow import deploy



_cfg: dict[str, Any] = {}
_flux_hf = _cfg.get("fluxHf") if isinstance(_cfg.get("fluxHf"), dict) else {}

FLUX2_GIT = str(_flux_hf.get("flux2Git") or "https://github.com/black-forest-labs/flux2.git")
FLUX2_DIR = str(_flux_hf.get("flux2Dir") or "/opt/flux2")

HF_KLEIN_KV = str(
    _flux_hf.get("kleinKvRepoId") or "black-forest-labs/FLUX.2-klein-9B-kv",
)
HF_DEV = str(_flux_hf.get("devRepoId") or "black-forest-labs/FLUX.2-dev")
HF_QWEN = str(_flux_hf.get("qwenRepoId") or "Qwen/Qwen3-8B-FP8")

KLEIN_KV_DIR = f"/models/{HF_KLEIN_KV}"
DEV_DIR = f"/models/{HF_DEV}"
QWEN_DIR = f"/models/{HF_QWEN}"
KLEIN_KV_WEIGHTS = f"{KLEIN_KV_DIR}/flux-2-klein-9b-kv.safetensors"
AE_WEIGHTS = f"{DEV_DIR}/ae.safetensors"

MODEL_KEY = "flux.2-klein-9b-kv"

_volume_name = str(_cfg.get("volumeName") or "models")
volume = modal.Volume.from_name(_volume_name, create_if_missing=True)

from tongflow.models.image_edit import ImageEditInput, ImageEditOutput
from tongflow.models.image_fusion import ImageFusionInput, ImageFusionOutput
from tongflow.node_slots import NodeSlots
from tongflow.protocol import asset, prompt_media_to_bytes
from tongflow.slots import node_slot

# ── app ──────────────────────────────────────────────────────────────────────

app = modal.App(Path(__file__).resolve().parent.name)

image = (
    # Use the *-devel image: the runtime variant has no gcc, and some deps (Triton /
    # kernel JIT) fail with ``Failed to find C compiler``. Same choice as gemma4.py.
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    .apt_install("git", "build-essential")
    .run_commands(f"git clone --depth 1 {FLUX2_GIT} {FLUX2_DIR}")
    .pip_install(
        "tongflow==0.1.0",
        "einops==0.8.1",
        "transformers==4.56.1",
        "safetensors>=0.4.5",
        "accelerate==1.12.0",
        "pillow>=10.0",
        "huggingface_hub>=0.25",
        "torchvision",
    )
    .env(
        {
            "PYTHONPATH": f"{FLUX2_DIR}/src",
            "HF_HOME": "/models/hf",
            # Reduce VRAM fragmentation (PyTorch's recommended setting)
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)

# Klein + Qwen3 TE + VAE all resident on GPU saturates 48GB; at inference time we follow the official
# CLI's CPU offload (flow model on CPU first, swap to GPU after text encoding). L40S is sufficient.
@deploy
@app.cls(
    scaledown_window=5,
    image=image,
    gpu="L40S",
    volumes={"/models": volume},
    timeout=1200,
)
class Inference:
    @staticmethod
    def _patch_local_weights():
        """If ``download`` already placed weights under ``/models/...``, use the local paths to avoid re-pulling from the Hub."""
        from flux2.text_encoder import Qwen3Embedder
        import flux2.util as util

        if os.path.isfile(KLEIN_KV_WEIGHTS):
            os.environ["KLEIN_9B_KV_MODEL_PATH"] = KLEIN_KV_WEIGHTS
        if os.path.isfile(AE_WEIGHTS):
            os.environ["AE_MODEL_PATH"] = AE_WEIGHTS
        if os.path.isfile(os.path.join(QWEN_DIR, "config.json")):

            def _te_klein(device="cuda"):
                return Qwen3Embedder(model_spec=QWEN_DIR, device=device)

            util.FLUX2_MODEL_INFO[MODEL_KEY]["text_encoder_load_fn"] = _te_klein

    @modal.enter()
    def load(self):
        import torch

        self._patch_local_weights()

        from flux2.util import FLUX2_MODEL_INFO, load_ae, load_flow_model, load_text_encoder

        self.torch_device = torch.device("cuda")
        self.model_info = FLUX2_MODEL_INFO[MODEL_KEY]
        self.text_encoder = load_text_encoder(MODEL_KEY, device=self.torch_device)
        # Load the flow model on CPU first; `_generate` moves it to GPU later to avoid co-resident VRAM with Qwen3.
        self.model = load_flow_model(MODEL_KEY, debug_mode=False, device="cpu")
        self.ae = load_ae(MODEL_KEY, device=self.torch_device)
        self.model.eval()
        self.ae.eval()
        self.text_encoder.eval()
        volume.commit()

    def _generate(
        self,
        prompt: str,
        pil_images: List[Any],
        width: int,
        height: int,
        seed: Optional[int],
    ) -> Any:
        import torch
        from einops import rearrange
        from PIL import Image
        from flux2.sampling import (
            batched_prc_img,
            batched_prc_txt,
            denoise,
            denoise_cached,
            encode_image_refs,
            get_schedule,
            scatter_ids,
        )

        dev = self.torch_device
        defaults = self.model_info.get("defaults", {})
        num_steps = defaults["num_steps"]
        guidance = defaults["guidance"]

        with torch.inference_mode():
            ref_tokens, ref_ids = encode_image_refs(self.ae, pil_images)

            ctx = self.text_encoder([prompt]).to(torch.bfloat16)
            ctx, ctx_ids = batched_prc_txt(ctx)

            # Match official scripts/cli.py cpu_offloading: after encoding the prompt, offload the text encoder, then move the flow model on.
            self.text_encoder.cpu()
            gc.collect()
            torch.cuda.empty_cache()
            self.model.to(dev)

            shape = (1, 128, height // 16, width // 16)
            gen_seed = seed if seed is not None else random.randrange(2**31)
            generator = torch.Generator(device="cuda").manual_seed(gen_seed)
            randn = torch.randn(
                shape, generator=generator, dtype=torch.bfloat16, device="cuda"
            )
            x, x_ids = batched_prc_img(randn)

            timesteps = get_schedule(num_steps, x.shape[1])
            use_kv = bool(self.model_info.get("use_kv_cache")) and (
                ref_tokens is not None and ref_ids is not None
            )
            if use_kv:
                x = denoise_cached(
                    self.model,
                    x,
                    x_ids,
                    ctx,
                    ctx_ids,
                    timesteps=timesteps,
                    guidance=guidance,
                    img_cond_seq=ref_tokens,
                    img_cond_seq_ids=ref_ids,
                )
            else:
                x = denoise(
                    self.model,
                    x,
                    x_ids,
                    ctx,
                    ctx_ids,
                    timesteps=timesteps,
                    guidance=guidance,
                    img_cond_seq=ref_tokens,
                    img_cond_seq_ids=ref_ids,
                )
            del ctx, ctx_ids, ref_tokens, ref_ids

            self.model.cpu()
            gc.collect()
            torch.cuda.empty_cache()

            x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
            x = self.ae.decode(x).float()
            x = x.clamp(-1, 1)
            x = rearrange(x[0], "c h w -> h w c")

            self.text_encoder.to(dev)

        return Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

    def _edit_to_png_bytes(
        self,
        prompt: str,
        image: bytes,
        seed: Optional[int] = None,
        match_input_size: bool = True,
        width: int = 1360,
        height: int = 768,
    ) -> bytes:
        from PIL import Image

        pil = Image.open(io.BytesIO(image)).convert("RGB")
        w, h = (pil.size[0], pil.size[1]) if match_input_size else (width, height)
        out = self._generate(prompt, [pil], w, h, seed)
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()

    def _edit_multi_to_png_bytes(
        self,
        prompt: str,
        images: List[bytes],
        seed: Optional[int] = None,
        width: int = 1360,
        height: int = 768,
    ) -> bytes:
        from PIL import Image

        pil_images = [Image.open(io.BytesIO(b)).convert("RGB") for b in images]
        out = self._generate(prompt, pil_images, width, height, seed)
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()

    @modal.method()
    def edit(
        self,
        prompt: str,
        image: bytes,
        seed: Optional[int] = None,
        match_input_size: bool = True,
        width: int = 1360,
        height: int = 768,
    ) -> bytes:
        return self._edit_to_png_bytes(
            prompt=prompt,
            image=image,
            seed=seed,
            match_input_size=match_input_size,
            width=width,
            height=height,
        )

    @modal.method()
    def edit_multi(
        self,
        prompt: str,
        images: List[bytes],
        seed: Optional[int] = None,
        width: int = 1360,
        height: int = 768,
    ) -> bytes:
        return self._edit_multi_to_png_bytes(
            prompt=prompt,
            images=images,
            seed=seed,
            width=width,
            height=height,
        )

    @modal.method()
    def text_to_image(
        self,
        prompt: str,
        seed: Optional[int] = None,
        width: int = 1360,
        height: int = 768,
    ) -> bytes:
        from PIL import Image

        out = self._generate(prompt, [], width, height, seed)
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()

    @modal.method()
    @node_slot(NodeSlots.IMAGE_EDIT)
    def image_edit(self, input: ImageEditInput) -> ImageEditOutput:
        if input.image is None:
            return ImageEditOutput(success=False, error="Missing image")
        raw = self._edit_to_png_bytes(
            prompt=input.text or "",
            image=prompt_media_to_bytes(input.image),
            seed=input.seed,
            width=input.width if input.width is not None else 1360,
            height=input.height if input.height is not None else 768,
            match_input_size=input.match_input_size
            if input.match_input_size is not None
            else True,
        )
        return ImageEditOutput(success=True, image=asset(raw, mime="image/png"))

    @modal.method()
    @node_slot(NodeSlots.IMAGE_FUSION)
    def image_fusion(self, input: ImageFusionInput) -> ImageFusionOutput:
        imgs = input.images or []
        if len(imgs) < 2:
            return ImageFusionOutput(success=False, error="Need at least 2 images")
        blobs: List[bytes] = [prompt_media_to_bytes(x) for x in imgs]
        raw = self._edit_multi_to_png_bytes(
            prompt=input.text or "",
            images=blobs,
            seed=input.seed,
            width=input.width if input.width is not None else 1360,
            height=input.height if input.height is not None else 768,
        )
        return ImageFusionOutput(success=True, image=asset(raw, mime="image/png"))
