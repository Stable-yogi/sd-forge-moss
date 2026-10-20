"""MOSS-Image (OpenMOSS) engine for Forge Neo: MOSS DiT + Qwen3.5-9B text model + FLUX.2 VAE."""

import torch

from backend import memory_management
from backend.args import dynamic_args
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE

from .text_engine import MOSSTextProcessingEngine


class MOSSImage(ForgeDiffusionEngine):
    matched_guesses = []  # filled by scripts/moss_register.py

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(model_dict={"qwen35_9b": huggingface_components["text_encoder"]}, tokenizer_dict={"qwen35_9b": huggingface_components["tokenizer"]})
        vae = VAE(model=huggingface_components["vae"], is_flux2=True)
        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None, k_predictor=self._get_predictor(), config=estimated_config)

        self.text_processing_engine_moss = MOSSTextProcessingEngine(
            text_encoder=clip.cond_stage_model.qwen35_9b,
            tokenizer=clip.tokenizer.qwen35_9b,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.use_shift = False  # shift fixed at 5.0 (ComfyUI MOSSImage sampling_settings)
        self.is_moss = True
        dynamic_args.moss = True  # read by the can_concat patch (cond / uncond lengths differ)

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        return self.text_processing_engine_moss(prompt)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        count = len(self.text_processing_engine_moss.tokenize([prompt])[0])
        return count, max(999, count)
