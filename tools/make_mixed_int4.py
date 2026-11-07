"""Build a "mixed" MOSS-Image file: INT4 everywhere, selected layers kept at INT8.

Why: a plain INT4 file of MOSS-Image can give watery, soft pictures. The four layer types of each block
are not equally sensitive. Keeping the `ffn_down` layers at INT8 gives pictures close to the full INT8 file
at about two thirds of its size (3.6 GB instead of 5.4 GB).

No GPU needed. Needs: pip install safetensors torch

Example (the default recipe, ffn_down of all 42 blocks at INT8):
    python make_mixed_int4.py --int4 moss_image_sft_int4_convrot.safetensors ^
        --int8 moss_image_sft_int8_convrot.safetensors --out moss_image_sft_int4_mixed_convrot.safetensors

Other recipes: --parts qkv,o,ffn_up_gate,ffn_down (which layer types) and --blocks 0-3,38-41 (which blocks).
Both inputs must be ConvRot files of the same model (for example both "sft").
"""
import argparse
import json
import re

from safetensors import safe_open
from safetensors.torch import save_file

PARTS = ("qkv", "o", "ffn_up_gate", "ffn_down")
NUM_BLOCKS = 42


def parse_blocks(text):
    if not text:
        return set(range(NUM_BLOCKS))
    out = set()
    for part in text.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--int4", required=True, help="the INT4 ConvRot file (the base)")
    ap.add_argument("--int8", required=True, help="the INT8 ConvRot file (gives the sensitive layers)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default="ffn_down", help="comma list of: " + ", ".join(PARTS))
    ap.add_argument("--blocks", default="", help="blocks to swap, e.g. 0-3,38-41 (default: all 42)")
    a = ap.parse_args()

    parts = [p for p in a.parts.split(",") if p]
    bad = [p for p in parts if p not in PARTS]
    if bad:
        raise SystemExit(f"unknown part(s): {bad}; choose from {PARTS}")
    blocks = parse_blocks(a.blocks)

    f4 = safe_open(a.int4, "pt")
    f8 = safe_open(a.int8, "pt")
    meta = dict(f4.metadata() or {})
    if "_quantization_metadata" not in meta:
        raise SystemExit("the INT4 file has no _quantization_metadata: not a ConvRot quantized file")
    qm = json.loads(meta["_quantization_metadata"])

    swap = sorted(f"layers.{n}.proj.{p}" for n in blocks for p in parts if f"layers.{n}.proj.{p}" in qm["layers"])
    if not swap:
        raise SystemExit("no matching layers found in the INT4 file")
    swap_set = set(swap)

    sd = {}
    for k in f4.keys():
        m = re.match(r"(layers\.\d+\.proj\.\w+)\.(weight|weight_scale)$", k)
        if m and m.group(1) in swap_set:
            continue
        sd[k] = f4.get_tensor(k)
    for name in swap:
        sd[name + ".weight"] = f8.get_tensor(name + ".weight")
        sd[name + ".weight_scale"] = f8.get_tensor(name + ".weight_scale")
        qm["layers"][name] = json.loads(bytes(f8.get_tensor(name + ".comfy_quant").tolist()).decode())
    meta["_quantization_metadata"] = json.dumps(qm)
    save_file(sd, a.out, metadata=meta)
    print(f"swapped {len(swap)} layers to INT8 -> {a.out}")


if __name__ == "__main__":
    main()
