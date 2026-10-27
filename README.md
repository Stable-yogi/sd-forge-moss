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

Forge Neo `neo-2.30.1` (branch `neo`, commit `ad60793`, 21 October 2026) and the newer Forge Neo with built-in MOSS-Image
(commit `c6c2b43`, 27 October 2026), with an RTX 5090 (32 GB):
MOSS-Image SFT int8 + Qwen3.5-9B int8, 1024 x 1024, 25 steps.

## If your Forge Neo already has MOSS-Image built in

Newer Forge Neo (from 21 October 2026) can run MOSS-Image without this extension. The extension notices this and
leaves Forge's own MOSS-Image code, preset and settings alone.
It still does one useful job: it finds the Qwen3.5 text encoder and the FLUX.2 VAE for you, so you only pick the checkpoint.
- **You want that help:** keep the extension. Update it (pull the latest from this page).
- **You pick the text encoder and VAE yourself:** you can remove the folder from `extensions/`.
- **You see "You do not have Qwen3.5 state dict!":** you have a MOSS-Image checkpoint but no text encoder picked.
  Update this extension, or pick the Qwen3.5 text encoder and the FLUX.2 VAE in the Forge module list.
- **You keep an older Forge:** keep the extension. It adds MOSS-Image to Forge by itself.

## Licence

Code: GPL-3.0 (see `LICENSE` and `NOTICE.md`). The model weights have their own licence from
OpenMOSS (Apache-2.0). This project is not made or endorsed by OpenMOSS.

## Spectrum speed-up (optional, off by default)

Open the accordion **MOSS Spectrum speed-up** and switch it on. The extension then skips about half of the model runs and forecasts them. On a 25-step picture this makes sampling about 2x faster. The picture is close to the normal one (same scene, same skin and freckles) but small details can differ, so test it on your own prompts. It works on the first (base) pass only, not on the hires pass.

- Default: 4 warm-up steps, 2 last steps, window 2, growth 0.75. Safer (about 1.6x, closer to normal): warm-up 6, last steps 3, growth 0.5, forecast mix 0.7.
- Tested on MOSS INT8 and INT6, 1024x1024, 25 steps.

Idea: Spectrum by Han et al. (arXiv:2603.01623, MIT). See `NOTICE.md`.
