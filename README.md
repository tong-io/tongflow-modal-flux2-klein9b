# tongflow-modal-flux2-klein9b

Official TongFlow plugin. Multi-reference image fusion and instruction-based image editing with **FLUX.2 Klein 9B** (`black-forest-labs/FLUX.2-klein-9b-kv`, with `FLUX.2-dev` and `Qwen/Qwen3-8B-FP8`), running on a GPU via [Modal](https://modal.com).

## Capabilities

- **Image fusion** (`image-fusion`) — blend or edit multiple reference images into one.
- **Image editing** (`image-edit`) — inpaint, edit, or redraw an image with instructions.

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `MODAL_TOKEN_ID` | ✅ | Create at [modal.com/settings/tokens](https://modal.com/settings/tokens). |
| `MODAL_TOKEN_SECRET` | ✅ | Paired with `MODAL_TOKEN_ID`. |

### Gated weights (Hugging Face)

The FLUX.2 weights are **gated**. Accept the FLUX license on Hugging Face for `black-forest-labs/FLUX.2-dev` and `black-forest-labs/FLUX.2-klein-9b-kv`, then create the Modal secret the plugin reads at deploy time:

```bash
modal secret create huggingface HF_TOKEN=hf_xxx
```

Without it, fetching the gated weights returns HTTP 403. (`Qwen/Qwen3-8B-FP8` is public.)
