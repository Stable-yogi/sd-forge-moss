# sd-forge-moss - MOSS-Image for Forge Neo

Adds text-to-image support for **MOSS-Image** (OpenMOSS) to
[Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo).
It runs inside Forge: the normal checkpoint list, Generate button and samplers.

## Files you need

From the ComfyUI package [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image):

| File | Size | Put it in |
|---|---|---|
| `moss_image_sft_int8_convrot.safetensors` (or `_sft_bf16`) | 5.4 GB / 10.2 GB | `models/Stable-diffusion` |
| `qwen3.5_9b_int8_convrot.safetensors` (or `_bf16`, text encoder) | 9.7 GB / 19.3 GB | `models/text_encoder` |
| `flux2-vae.safetensors` | 0.3 GB | `models/VAE` |

You only pick the checkpoint. The extension finds the text encoder (a file with `qwen3.5_9b` in its name)
and the VAE (a file with `flux2` and `vae` in its name) by itself.
On a card with more than 40 GB it picks the bf16 text encoder first, else the int8 one.

## Use

1. Copy this folder into `extensions/` (or *Extensions > Install from URL*). Restart Forge fully.
2. Pick the UI preset **moss**.
3. Pick the MOSS-Image checkpoint.
4. The preset sets the settings: **Euler, Simple, 25 steps, CFG 2.5, 1024 x 1024**.

## Notes

- Text-to-image only. Shift is fixed at 5.0.
- The `base` model loads too; the `sft` model gives the better pictures.

## Tested on

Forge Neo `neo-2.30.1` (branch `neo`, commit `ad60793`, 21 October 2026), with an RTX 5090 (32 GB):
MOSS-Image SFT int8 + Qwen3.5-9B int8, 1024 x 1024, 25 steps.

## When Forge Neo adds MOSS-Image itself

The Forge Neo maker is working on built-in MOSS-Image support.
- **You update Forge to a version with built-in MOSS:** the extension sees it and switches itself off.
  You can then remove the folder from `extensions/`. Use Forge's own MOSS preset.
- **You keep an older Forge:** keep the extension. It keeps working; you do not have to update Forge.

## Licence

Code: GPL-3.0 (see `LICENSE` and `NOTICE.md`). The model weights have their own licence from
OpenMOSS (Apache-2.0). This project is not made or endorsed by OpenMOSS.
