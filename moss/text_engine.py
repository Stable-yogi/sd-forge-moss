"""MOSS-Image prompt conditioning on the Qwen3.5-9B text model (ComfyUI 0.42.1 moss_image.py)."""

import re

import torch

from backend import memory_management

_CONTROL_TAG = re.compile(r"<(?:lora|lyco|l2):[^<>]*>", re.IGNORECASE)


def strip_non_text_tags(text: str) -> str:
    """Keep Forge control syntax out of the text conditioning."""
    return _CONTROL_TAG.sub("", text).strip()


# chat template with enable_thinking=True: trimmed user turn, open think block
MOSS_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n"


class MOSSTextProcessingEngine:
    def __init__(self, text_encoder, tokenizer):
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

    def tokenize(self, texts):
        out = []
        for text in texts:
            text = strip_non_text_tags(text)
            llama_text = text if text.startswith("<|im_start|>") else MOSS_TEMPLATE.format(text.strip())
            out.append(self.tokenizer(llama_text, add_special_tokens=False)["input_ids"])
        return out

    def __call__(self, texts):
        device = memory_management.text_encoder_device()
        outputs = []
        cache = {}
        for text, ids in zip(texts, self.tokenize(texts)):
            if text not in cache:
                tokens = torch.tensor([ids], device=device, dtype=torch.long)
                cache[text] = self.text_encoder(tokens)[0]
            outputs.append(cache[text])
        return outputs
