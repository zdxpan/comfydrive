import os
import random
import torch
import numpy as np

from nodes import NODE_CLASS_MAPPINGS
from comfyui_workflows.comfyui_tools import get_value_at_index
from src.utils.aliyun_v2 import AliyunOss
from comfyui_workflows.utils import ServiceTemplate
from folder_paths import base_path
from comfyui_workflows.refiner_v1 import Refinerv1
from comfyui_workflows.zdx_nodes import ObjExtractByMask
from comfyui_workflows.utils import get_or_download_image


class F1FillReplace():
    @torch.inference_mode
    def __init__(self):
        # 1、transformer and lora
        self.unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        self.unetloader_820 = self.unetloader.load_unet(
            unet_name="FLUX.1-Fill-dev_fp8.safetensors", weight_dtype="fp8_e4m3fn"
        )
        self.loraloader = NODE_CLASS_MAPPINGS["LoraLoader"]()
        self.loraloader_826 = self.loraloader.load_lora(
            lora_name="FLUX.1-Turbo-Alpha.safetensors",
            strength_model=1,
            strength_clip=1,
            model=get_value_at_index(self.unetloader_820, 0),
            clip=get_value_at_index(self.dualcliploader_822, 0),
        )
        # 2、clip & t5 texn_encoder
        self.dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        self.dualcliploader_822 = self.dualcliploader.load_clip(
            clip_name1="t5xxl_fp8_e4m3fn.safetensors",
            clip_name2="clip_l.safetensors",
            type="flux",
        )
        self.cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()

        # 3、vae 
        self.vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        self.vaeloader_821 = self.vaeloader.load_vae(vae_name="ae.safetensors")


        # 4/load redux model  & vision encoder  & fill_redux style mode
        self.clipvisionloader = NODE_CLASS_MAPPINGS["CLIPVisionLoader"]()
        self.clipvisionloader_329 = self.clipvisionloader.load_clip(
            clip_name="sigclip_vision_patch14_384.safetensors"
        )
        self.clipvisionencode = NODE_CLASS_MAPPINGS["CLIPVisionEncode"]()
        self.cliptextencodeflux = NODE_CLASS_MAPPINGS["CLIPTextEncodeFlux"]()

        self.stylemodelloader = NODE_CLASS_MAPPINGS["StyleModelLoader"]()
        self.stylemodelloader_330 = self.stylemodelloader.load_style_model(
            style_model_name="flux1-redux-dev.safetensors"
        )
        self.differentialdiffusion = NODE_CLASS_MAPPINGS["DifferentialDiffusion"]()
        self.stylemodelapply = NODE_CLASS_MAPPINGS["StyleModelApply"]()

        # 4.1  final flux fill model 
        self.differentialdiffusion_327 = self.differentialdiffusion.apply(
            model=get_value_at_index(self.loraloader_439, 0)
        )
        
        # 5/sampler and  vae decode
        self.ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        self.modelsamplingflux = NODE_CLASS_MAPPINGS["ModelSamplingFlux"]()

        self.fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
        self.vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()

        # - - fix prompt conds - - ~
        self.cond = self.dualcliploader_822
        cliptextencodeflux_323 = self.cliptextencodeflux.encode(
            clip_l="", t5xxl="", guidance=30, clip=get_value_at_index(self.loraloader_439, 1)
        )
        cliptextencodeflux_325 = self.cliptextencodeflux.encode(
            clip_l="", t5xxl="", guidance=30, clip=get_value_at_index(self.loraloader_439, 1)
        )

    @torch.inference_mode
    def __call__(self, opt):
        input_obj_image = opt.obj_image   # rembg ,final image input
        input_target_image = opt.target_image
        orig_w,orig_h = input_target_image.size
        obj_target_concated_im = cat(input_obj_image, input_target_image)
        width, height = obj_target_concated_im.size
        # canvas and mask load~
        # obj       + image_canvas
        # sold_mask + image_targe_mm
        obj_solid_mask = torch.zeros_like(input_obj_image)
        target_mask = opt.target_mask
        obj_target_concated_mask = cat(obj_solid_mask, target_mask)
        clipvisionencode_172 = self.clipvisionencode.encode(
            clip_vision=get_value_at_index(self.clipvisionloader_329, 0),
            image=input_obj_image,
        )
        self.cleangpu(self.clipvisionloader_329)  # clear memory of clip_version~


        # conditiong
        cliptextencodeflux_323 = self.cliptextencodeflux.encode(clip_l="", t5xxl="", guidance=30, clip=get_value_at_index(self.loraloader_439, 1))
        cliptextencodeflux_325 = self.cliptextencodeflux.encode(clip_l="", t5xxl="", guidance=30, clip=get_value_at_index(self.loraloader_439, 1))

        # concated obj_img and mask make conditiong and generate an Latent!!! like image 2 image
        inpaintmodelconditioning_220 = self.inpaintmodelconditioning.encode(
            positive=get_value_at_index(cliptextencodeflux_323, 0),
            negative=get_value_at_index(cliptextencodeflux_325, 0),
            vae=get_value_at_index(self.vaeloader, 0),
            pixels=obj_target_concated_im,
            mask=obj_target_concated_mask,
        )

        modelsamplingflux_320 = self.modelsamplingflux.patch(
            max_shift=1.1500000000000001,
            base_shift=0.5,
            width=width,
            height=height,
            model=get_value_at_index(self.differentialdiffusion_327, 0),
        )

        stylemodelapply_171 = self.stylemodelapply.apply_stylemodel(
            conditioning=get_value_at_index(inpaintmodelconditioning_220, 0),
            style_model=get_value_at_index(self.stylemodelloader_330, 0),
            clip_vision_output=get_value_at_index(clipvisionencode_172, 0),
        )

        fluxguidance_223 = self.fluxguidance.append(
            guidance=30, conditioning=get_value_at_index(stylemodelapply_171, 0)
        )

        ksampler_102 = self.ksampler.sample(
            seed=random.randint(1, 2**64),
            steps=8,
            cfg=1,
            sampler_name="euler",
            scheduler="beta",
            denoise=1,
            model=get_value_at_index(modelsamplingflux_320, 0),
            positive=get_value_at_index(fluxguidance_223, 0),
            negative=get_value_at_index(inpaintmodelconditioning_220, 1),
            latent_image=get_value_at_index(inpaintmodelconditioning_220, 2),
        )

        vaedecode_106 = self.vaedecode.decode(
            samples=get_value_at_index(ksampler_102, 0),
            vae=get_value_at_index(self.vaeloader, 0),
        )

        easy_imagesplitgrid_294 = self.easy_imagesplitgrid.doit(
            row=1, column=2, images=get_value_at_index(vaedecode_106, 0)
        )

        easy_imagessplitimage_299 = self.easy_imagessplitimage.split(
            images=get_value_at_index(easy_imagesplitgrid_294, 0)
        )

        refiner_res = self.refiner.forward((get_value_at_index(easy_imagessplitimage_299, 1)))

        # final res image 
        res_im = tensor2image(
            get_value_at_index(easy_imagessplitimage_299, 1)
        )
        return res_im
