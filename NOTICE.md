# Notice

**Code licence: GPL-3.0** (full text in `LICENSE`).
The MOSS-Image model code in `moss/` is a port of ComfyUI 0.42.1 (`comfy/ldm/moss_image/model.py`,
the Qwen3.5 text model and the MOSS-Image text encoder, GPL-3.0, Comfy Org). The port is modified:
it uses Forge Neo's own layer helpers, attention and sampler hooks.

**No model weights are in this folder.** The MOSS-Image weights come from OpenMOSS and have their own
licence (Apache-2.0). Read and follow it before you use or share the model.

This extension is not made, checked or endorsed by OpenMOSS.

Config files in `hf_config/MOSS-Image/` (text encoder, tokenizer, VAE, scheduler, transformer
configs) are copied from the public MOSS-Image / Qwen3.5 / FLUX.2 repos.

## Spectrum speed-up (optional)

`moss/spectrum.py` and `scripts/moss_spectrum.py` use the idea of **Spectrum** (Han et al., "Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration", arXiv:2603.01623, MIT licence, https://github.com/hanjq17/Spectrum). The code is our own, written from the paper. We also thank the author of the ComfyUI node ComfyUI-Spectrum-MOSS (GPL-3.0, https://github.com/Rkkss/ComfyUI-Spectrum-MOSS) for showing that the idea works on MOSS; none of its code was copied.
