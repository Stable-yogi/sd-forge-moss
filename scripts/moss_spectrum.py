"""Optional MOSS speed-up: Spectrum step forecast (off by default). Logic lives in moss/spectrum.py."""
import gradio as gr

from modules import scripts
from modules.ui_components import InputAccordion

try:
    from moss import spectrum as S
except Exception:  # extension folder name differs: load by path
    import importlib.util
    import os

    _p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "moss", "spectrum.py")
    _sp = importlib.util.spec_from_file_location("moss_spectrum_impl", _p)
    S = importlib.util.module_from_spec(_sp)
    _sp.loader.exec_module(S)


class MossSpectrum(scripts.Script):
    sorting_priority = 16000

    def title(self):
        return "MOSS Spectrum speed-up"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(False, label="MOSS Spectrum speed-up (faster, a little less exact)") as enabled:
            gr.HTML("Skips most model runs and forecasts them. About 2x faster on MOSS. Test it on your own prompts first.")
            with gr.Row():
                warmup = gr.Slider(1, 10, value=4, step=1, label="Warm-up real steps")
                tail = gr.Slider(0, 6, value=2, step=1, label="Last real steps")
            with gr.Row():
                window = gr.Slider(1.0, 4.0, value=2.0, step=0.25, label="Window (1 real step in every N)")
                flex = gr.Slider(0.0, 2.0, value=0.75, step=0.05, label="Window growth")
            with gr.Row():
                degree = gr.Slider(1, 6, value=4, step=1, label="Polynomial degree")
                lam = gr.Slider(0.0, 1.0, value=0.1, step=0.01, label="Ridge strength")
                blend = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Forecast mix (1 = only polynomial)")
        self.infotext_fields = []
        return [enabled, warmup, tail, window, flex, degree, lam, blend]

    def process_before_every_sampling(self, p, *args, **kwargs):
        enabled, warmup, tail, window, flex, degree, lam, blend = args[:8]
        if not enabled or getattr(p, "is_hr_pass", False):
            return
        unet = p.sd_model.forge_objects.unet
        dm = getattr(getattr(unet, "model", None), "diffusion_model", None)
        if dm is None or not hasattr(dm, "layers") or not hasattr(dm, "final_linear_video"):
            print("[moss-spectrum] this model is not MOSS-Image; Spectrum is off.")
            return
        steps = int(p.steps)
        st = S.Spectrum(steps, warmup, window, flex, tail, degree, lam, blend)
        unet = unet.clone()
        if unet.model_options.get("model_function_wrapper") is not None:
            print("[moss-spectrum] another extension already wraps the model call; Spectrum replaces it for this run.")
        unet.set_model_unet_function_wrapper(S.make_wrapper(st))
        p.sd_model.forge_objects.unet = unet
        self._state = st
        p.extra_generation_params["MOSS Spectrum"] = "w%s,win%s,flex%s,tail%s,deg%s,lam%s,mix%s" % (warmup, window, flex, tail, degree, lam, blend)
        print("[moss-spectrum] on: %d of %d steps run the model" % (sum(st.plan), steps))

    def postprocess(self, p, processed, *args):
        st = getattr(self, "_state", None)
        if st is not None:
            print("[moss-spectrum] " + st.summary())
            self._state = None
