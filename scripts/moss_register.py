"""
MOSS-Image registration - runs on Forge load. Adds the MOSS-Image architecture to Forge Neo
at runtime (no core edits):
  * detection fingerprint + a model_list entry (MOSSImageBase)
  * the loader's builders (Qwen3.5-9B text model, MOSS DiT)
  * the text encoder key mapping (Qwen3.5 file -> qwen35_9b.transformer.*)
  * the diffusion engine (moss.engine.MOSSImage)
  * cond / uncond of different text lengths never get repeat-padded together
  * a bare DiT auto-loads its text encoder + VAE
  * a 'moss' UI preset (Euler / Simple, 25 steps, CFG 2.5, 1024x1024)

Every step is wrapped so a failure never breaks Forge boot. If Forge ever ships MOSS-Image natively,
this extension steps aside.
"""
import json
import os
import struct
import sys
import traceback

EXT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # extensions/sd-forge-moss
if EXT_ROOT not in sys.path:
    sys.path.insert(0, EXT_ROOT)
CFG_DIR = os.path.join(EXT_ROOT, "hf_config", "MOSS-Image")

# Fixed time shift from ComfyUI MOSSImage sampling_settings.
MOSS_SHIFT = 5.0

PRESET = "moss"
MOSS_PRESET_DEFAULTS = {
    "moss_t2i_sampler": "Euler", "moss_i2i_sampler": "Euler",
    "moss_t2i_scheduler": "Simple", "moss_i2i_scheduler": "Simple",
    "moss_t2i_step": 25, "moss_t2i_hr_step": 25, "moss_i2i_step": 25,
    "moss_t2i_cfg": 2.5, "moss_t2i_hr_cfg": 2.5, "moss_i2i_cfg": 2.5,
    "moss_t2i_width": 1024, "moss_t2i_height": 1024,
    "moss_i2i_width": 1024, "moss_i2i_height": 1024,
    "moss_t2i_batch_size": 1, "moss_i2i_batch_size": 1,
}
# opts.set() needs these in data_labels (KeyError otherwise)
MOSS_HIDDEN_OPTS = {
    "forge_checkpoint_moss": None,
    "forge_additional_modules_moss": [],
    "forge_unet_storage_dtype_moss": "Automatic",
}

TE_KEY = "model.language_model.layers.0.linear_attn.A_log"
_EXTS = (".safetensors", ".sft", ".gguf")


def log(msg):
    print("[moss] " + msg)


def _transformer_cls():
    """The DiT class name the pipeline config uses (the loader keys its builder on it)."""
    with open(os.path.join(CFG_DIR, "model_index.json"), encoding="utf-8") as f:
        return json.load(f)["transformer"][1]


# ---------------------------------------------------------------- file sniffing
def _st_keys(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return [k for k in json.loads(f.read(n)).keys() if k != "__metadata__"]


def _is_moss_keys(keys):
    return any(k.endswith("shared_block_modulation.weight") for k in keys) and any(k.endswith("layers.0.proj.ffn_up_gate.weight") for k in keys)


def _bare_moss_dit(path):
    """True if the file IS a MOSS DiT and carries no text encoder / VAE."""
    try:
        p = str(path)
        low = p.lower()
        if low.endswith((".safetensors", ".sft")):
            keys = _st_keys(p)
            return _is_moss_keys(keys) and not any(k.startswith("text_encoders.") for k in keys)
        if low.endswith(".gguf"):
            with open(p, "rb") as f:
                head = f.read(16 << 20)
            return b"shared_block_modulation" in head and b"ffn_up_gate" in head and b"text_encoders." not in head
    except Exception:
        return False
    return False


def _is_flux2_vae(keys):
    return "decoder.post_quant_conv.weight" in keys or "bn.running_mean" in keys


def _big_gpu():
    try:
        import torch

        return torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory > 40 * 1024**3
    except Exception:
        return False


def _find_modules():
    """Find the Qwen3.5-9B text encoder and the FLUX.2 VAE in Forge's module folders."""
    from modules import shared

    try:
        from modules import paths

        root = paths.models_path
    except Exception:
        root = os.path.join(os.getcwd(), "models")
    co = getattr(shared, "cmd_opts", None)
    te_dirs = [os.path.join(root, "text_encoder")] + list(getattr(co, "text_encoder_dirs", []) or [])
    vae_dirs = [os.path.join(root, "VAE")] + list(getattr(co, "vae_dirs", []) or [])

    def files(dirs):
        out = []
        for d in dirs:
            try:
                out += [os.path.join(d, n) for n in sorted(os.listdir(d)) if n.lower().endswith(_EXTS)]
            except Exception:
                continue
        return out

    def pick(cands, prefer):
        for pk in prefer:
            for c in cands:
                if pk in os.path.basename(c).lower():
                    return c
        return cands[0] if cands else None

    # bf16 needs about 18 GB; a smaller card gets the int8 file first
    te_c = [f for f in files(te_dirs) if any(s in os.path.basename(f).lower() for s in ("qwen3.5_9b", "qwen3.5-9b", "qwen35_9b")) and "gptq" not in os.path.basename(f).lower()]
    te = pick(te_c, ("bf16", "int8") if _big_gpu() else ("int8", "bf16"))
    vae_c = [f for f in files(vae_dirs) if "flux2" in os.path.basename(f).lower() and "vae" in os.path.basename(f).lower()]
    vae = pick(vae_c, ("flux2-vae", "flux2_vae"))
    if vae is None:  # renamed file: look inside
        for f in files(vae_dirs):
            try:
                if f.lower().endswith((".safetensors", ".sft")) and _is_flux2_vae(set(_st_keys(f))):
                    vae = f
                    break
            except Exception:
                continue
    return te, vae


def _native_moss():
    try:
        import backend.loader as _l

        return any(getattr(M, "__name__", "") in ("MOSSImage", "MOSS") and not getattr(M, "_sd_forge_moss", False) for M in getattr(_l, "possible_models", ()))
    except Exception:
        return False


# ---------------------------------------------------------------- architecture
def _register_arch():
    import torch

    import backend.loader as loader
    import huggingface_guess.detection as detection
    from huggingface_guess import latent, model_list

    from moss.engine import MOSSImage

    if getattr(loader, "_sd_forge_moss", False):
        return

    class MOSSImageBase(model_list.BASE):
        huggingface_repo = CFG_DIR
        unet_config = {"image_model": "moss_image"}
        sampling_settings = {"multiplier": 1.0, "shift": MOSS_SHIFT}
        memory_usage_factor = 14.6
        unet_extra_config = {}
        latent_format = latent.Flux2
        supported_inference_dtypes = [torch.bfloat16, torch.float32]
        vae_key_prefix = ["vae."]
        text_encoder_key_prefix = ["text_encoders."]
        unet_target = "transformer"

        def clip_target(self, state_dict={}):
            return {"qwen35_9b.transformer": "text_encoder"}

    model_list.MOSSImageBase = MOSSImageBase
    model_list.models.insert(0, MOSSImageBase)

    # 1. detection fingerprint (checked before every native rule)
    _orig_detect = detection.detect_unet_config

    def _detect(state_dict, key_prefix):
        mod = "{}shared_block_modulation.weight".format(key_prefix)
        up = "{}layers.0.proj.ffn_up_gate.weight".format(key_prefix)
        if mod in state_dict and up in state_dict:
            keys = list(state_dict.keys())
            dim = int(state_dict[mod].shape[0]) // 6
            return {
                "image_model": "moss_image",
                "dim": dim,
                "ffn_dim": int(state_dict[up].shape[0]) // 2,
                "num_heads": dim // 128,
                "num_layers": detection.count_blocks(keys, "{}layers.".format(key_prefix) + "{}."),
                "fused_qkv": "{}layers.0.proj.qkv.weight".format(key_prefix) in state_dict,
            }
        return _orig_detect(state_dict, key_prefix)

    detection.detect_unet_config = _detect

    # 2. builders. The DiT reuses Forge's own build path (quant / fp8 / gguf handling)
    #    with our class swapped in for the duration of the call.
    dit_cls = _transformer_cls()
    _orig_lhc = loader.load_huggingface_component

    def _lhc(guess, component_name, lib_name, cls_name, repo_path, state_dict):
        if cls_name == "Qwen35TextModel":
            return _build_text_model(loader, os.path.join(repo_path, component_name), state_dict)
        if cls_name != dit_cls:
            return _orig_lhc(guess, component_name, lib_name, cls_name, repo_path, state_dict)
        import backend.nn.krea as krea_nn

        from moss.dit import MOSSImage as MOSSImageDiT

        if isinstance(state_dict, dict):
            state_dict.pop("rope.bands", None)
        native = krea_nn.SingleStreamDiT
        krea_nn.SingleStreamDiT = MOSSImageDiT
        try:
            return _orig_lhc(guess, component_name, lib_name, "Krea2Transformer2DModel", repo_path, state_dict)
        finally:
            krea_nn.SingleStreamDiT = native

    loader.load_huggingface_component = _lhc

    # 3. text encoder keys: Qwen3.5 file -> text_encoders.qwen35_9b.transformer.*
    _orig_replace = loader.replace_state_dict

    def _replace(sd, asd, guess, path):
        if TE_KEY in asd:
            prefix = guess.text_encoder_key_prefix[0]
            for k, v in asd.items():
                sd[f"{prefix}qwen35_9b.transformer.{k}"] = v
            return sd
        return _orig_replace(sd, asd, guess, path)

    loader.replace_state_dict = _replace

    # 4. engine
    MOSSImage.matched_guesses = [MOSSImageBase]
    MOSSImage._sd_forge_moss = True
    loader.possible_models = (MOSSImage,) + tuple(loader.possible_models)

    # 5. cond / uncond with different text lengths run apart (the DiT attends to every token)
    from backend.args import dynamic_args
    from backend.sampling import condition

    _orig_can_concat = condition.ConditionCrossAttn.can_concat

    def _can_concat(self, other):
        if getattr(dynamic_args, "moss", False) and self.cond.shape != other.cond.shape:
            return False
        return _orig_can_concat(self, other)

    condition.ConditionCrossAttn.can_concat = _can_concat

    loader._sd_forge_moss = True
    log("MOSS-Image architecture registered (shift %.1f)." % MOSS_SHIFT)


def _build_text_model(loader, config_path, state_dict):
    """Qwen3.5-9B text model, built the way Forge builds its other LLM text encoders."""
    import torch

    from moss.qwen35_text import Qwen35TextModel

    mm = loader.memory_management
    assert isinstance(state_dict, dict) and len(state_dict) > 16, "You do not have Qwen3.5 state dict!"
    config = loader.read_arbitrary_config(config_path)
    state_dict = {k.replace("model.language_model.", "model.", 1): v for k, v in state_dict.items() if k.startswith("model.language_model.")}

    storage_dtype = mm.text_encoder_dtype()
    state_dict_dtype = loader.utils.weight_dtype(state_dict)
    quant_config = loader.detect_quantization(state_dict)
    if quant_config is not None:
        storage_dtype = state_dict_dtype
        log("Qwen3.5 text model: mixed precision")
    elif state_dict_dtype in [torch.float8_e4m3fn, torch.float8_e5m2, "nf4", "fp4", "gguf"]:
        storage_dtype = state_dict_dtype

    if storage_dtype in ["nf4", "fp4", "gguf"]:
        with loader.no_init_weights():
            with loader.using_forge_operations(device=mm.cpu, dtype=mm.text_encoder_dtype(), manual_cast_enabled=False, extra_dtype=storage_dtype):
                model = Qwen35TextModel(config)
    else:
        with loader.no_init_weights():
            with loader.using_forge_operations(device=mm.cpu, dtype=storage_dtype, manual_cast_enabled=True, extra_dtype=quant_config):
                model = Qwen35TextModel(config)

    loader.load_state_dict(model, state_dict, log_name="Qwen35TextModel")
    return model


def _register_loader_wrapper(native=False):
    """Reset the moss flag per load; a bare MOSS DiT auto-loads its text encoder + VAE.
    native=True (Forge has MOSS-Image built in): only the auto-load, Forge keeps its own flag."""
    import backend.loader as loader
    import modules.sd_models as sdm
    from backend.args import dynamic_args

    if getattr(sdm.forge_loader, "_moss", False):
        return
    _orig = sdm.forge_loader

    def _forge_loader(sd, additional_state_dicts=None):
        if not native:
            dynamic_args.moss = False
        try:
            if _bare_moss_dit(sd):
                asd = list(additional_state_dicts or [])
                names = [os.path.basename(str(p)).lower() for p in asd]
                te, vae = _find_modules()
                add = []
                if not any("qwen3.5" in n or "qwen35" in n for n in names):
                    if te:
                        add.append(te)
                    else:
                        log("no Qwen3.5-9B text encoder found in the text encoder folders.")
                if not any("vae" in n for n in names):
                    if vae:
                        add.append(vae)
                    else:
                        log("no FLUX.2 VAE found in the VAE folders.")
                if add:
                    additional_state_dicts = add + asd
                    log("bare DiT -> auto-loaded: " + ", ".join(os.path.basename(m) for m in add))
        except Exception:
            log("auto module pick failed:\n" + traceback.format_exc())
        return _orig(sd, additional_state_dicts)

    _forge_loader._moss = True
    sdm.forge_loader = _forge_loader
    loader.forge_loader = _forge_loader


# ---------------------------------------------------------------- UI preset
def _register_preset(native=False):
    """native=True: Forge already has the moss preset and its own settings; only pre-fill the modules."""
    import modules_forge.presets as P

    if native:
        pass
    elif getattr(P.PresetArch.choices, "_moss", False) is False:
        _orig_choices = P.PresetArch.choices

        def _choices():
            c = list(_orig_choices())
            if PRESET not in c:
                c.append(PRESET)
            return c

        _choices._moss = True
        P.PresetArch.choices = staticmethod(_choices)

    try:
        if native:
            raise StopIteration
        from modules import shared

        try:
            from modules.options import OptionInfo
        except Exception:
            from modules.shared import OptionInfo
        for k, default in MOSS_HIDDEN_OPTS.items():
            if k not in shared.opts.data_labels:
                oi = OptionInfo(default)
                oi.section = (None, "Forge Hidden Options")
                shared.opts.data_labels[k] = oi
            shared.opts.data.setdefault(k, default)
        for k, v in MOSS_PRESET_DEFAULTS.items():
            shared.opts.data.setdefault(k, v)
    except StopIteration:
        pass
    except Exception:
        log("preset options step skipped:\n" + traceback.format_exc())

    import modules_forge.main_entry as ME

    if getattr(ME.on_preset_change, "_moss", False):
        return
    _orig = ME.on_preset_change

    def _wrapped(preset):
        if preset == PRESET:
            try:
                from modules import shared

                if not native:
                    for k, v in MOSS_PRESET_DEFAULTS.items():
                        shared.opts.data[k] = v
                    for k, v in MOSS_HIDDEN_OPTS.items():
                        shared.opts.data.setdefault(k, v)
                if not (shared.opts.data.get("forge_additional_modules_moss") or []):
                    auto = [m for m in _find_modules() if m]
                    if auto:
                        shared.opts.data["forge_additional_modules_moss"] = auto
            except Exception:
                log("preset pre-fill error:\n" + traceback.format_exc())
        return _orig(preset)

    _wrapped._moss = True
    ME.on_preset_change = _wrapped


if _native_moss():
    log("this Forge has MOSS-Image built in; the extension only finds the text encoder and VAE for you.")
    try:
        _register_loader_wrapper(native=True)
    except Exception:
        log("loader wrapper skipped:\n" + traceback.format_exc())
    try:
        _register_preset(native=True)
    except Exception:
        log("preset pre-fill skipped:\n" + traceback.format_exc())
else:
    try:
        _register_arch()
    except Exception:
        log("architecture registration FAILED (Forge boot unaffected):\n" + traceback.format_exc())
    try:
        _register_loader_wrapper()
    except Exception:
        log("loader wrapper skipped:\n" + traceback.format_exc())
    try:
        _register_preset()
    except Exception:
        log("preset registration skipped:\n" + traceback.format_exc())
