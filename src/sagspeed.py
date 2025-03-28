# must run at custom_nodes after custom nodes loaded!
from src.util import (
    find_path, add_comfyui_directory_to_sys_path,
    tensor2pil, pil2tensor, pilmask2tensor, make_image_grid, draw_text
)

from nodes import NODE_CLASS_MAPPINGS


add_comfyui_directory_to_sys_path()

from comfy.model_patcher import ModelPatcher

class zdxApplySageAtt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "use_SageAttention": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "zdx"
    TITLE = "zdx sage speed"

    def __init__(self):
        self.orig_attn = None

    def patch(self, model: ModelPatcher, use_SageAttention: bool):
        try:
            from comfy.ldm.flux import math

            if use_SageAttention:
                from sageattention import sageattn
                from comfy.ldm.modules.attention import attention_sage
                from comfy.ldm.modules import attention

                self.orig_attn = getattr(math, "optimized_attention")
                setattr(attention, "sageattn", sageattn)
                setattr(math, "optimized_attention", attention_sage)
            elif self.orig_attn is not None:
                setattr(math, "optimized_attention", self.orig_attn)
        except:
            pass

        return (model,)

class MultiplySigmas:
    # sigmas: input  
    # factor 0 ~ 100   "step": 0.001  default - 1.0
    # start:  0, 0~1  0.001  same with end
    FUNCTION = "simple_output"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/sigmas"

    def simple_output(self, sigmas, factor, start, end):
        # Clone the sigmas to ensure the input is not modified (stateless)
        # sigmas = sigmas.clone()
        
        total_sigmas = len(sigmas)
        start_idx = int(start * total_sigmas)
        end_idx = int(end * total_sigmas)

        for i in range(start_idx, end_idx):
            sigmas[i] *= factor
        return (sigmas,)

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import random
import os


# Schedule creation function from https://github.com/muerrilla/sd-webui-detail-daemon
def make_detail_daemon_schedule(
    steps,
    start,
    end,
    bias,
    amount,
    exponent,
    start_offset,
    end_offset,
    fade,
    smooth,
):
    start = min(start, end)
    mid = start + bias * (end - start)
    multipliers = np.zeros(steps)

    start_idx, mid_idx, end_idx = [
        int(round(x * (steps - 1))) for x in [start, mid, end]
    ]

    start_values = np.linspace(0, 1, mid_idx - start_idx + 1)
    if smooth:
        start_values = 0.5 * (1 - np.cos(start_values * np.pi))
    start_values = start_values**exponent
    if start_values.any():
        start_values *= amount - start_offset
        start_values += start_offset

    end_values = np.linspace(1, 0, end_idx - mid_idx + 1)
    if smooth:
        end_values = 0.5 * (1 - np.cos(end_values * np.pi))
    end_values = end_values**exponent
    if end_values.any():
        end_values *= amount - end_offset
        end_values += end_offset

    multipliers[start_idx : mid_idx + 1] = start_values
    multipliers[mid_idx : end_idx + 1] = end_values
    multipliers[:start_idx] = start_offset
    multipliers[end_idx + 1 :] = end_offset
    multipliers *= 1 - fade

    return multipliers




