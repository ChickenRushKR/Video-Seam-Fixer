"""Feasibility probe: which Wan pipeline loads the cached Wan2.2-TI2V-5B, and its call signature."""

import inspect
import torch

MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

for cls_name in ["WanImageToVideoPipeline", "WanPipeline"]:
    try:
        import diffusers

        cls = getattr(diffusers, cls_name)
        print(f"\n=== trying {cls_name} ===", flush=True)
        pipe = cls.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
        sig = inspect.signature(pipe.__call__)
        params = [p for p in sig.parameters if p not in ("self",)]
        print(f"LOADED {cls_name}; __call__ params: {params}", flush=True)
        print("components:", [k for k in pipe.components], flush=True)
        break
    except Exception as e:
        print(f"{cls_name} FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
