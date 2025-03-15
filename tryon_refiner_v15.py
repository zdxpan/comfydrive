import os
import time
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import tqdm, time, os
from PIL import Image
from src.util import (
    tensor2pil, pil2tensor, pilmask2tensor, make_image_grid, draw_text
)

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("comfyui")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import NODE_CLASS_MAPPINGS

class Refinerv1():
    def __init__(self):
        self.name = self.__class__.__name__
        with torch.inference_mode():
            self.clipvisionloader = NODE_CLASS_MAPPINGS["CLIPVisionLoader"]()
            self.imageresizekj = NODE_CLASS_MAPPINGS["ImageResizeKJ"]()
            self.imagescaleby = NODE_CLASS_MAPPINGS["ImageScaleBy"]()

            self.imageupscalewithmodel = NODE_CLASS_MAPPINGS["ImageUpscaleWithModel"]()
            
            self.loraloader = NODE_CLASS_MAPPINGS["LoraLoader"]()
            self.vaeencodetiled = NODE_CLASS_MAPPINGS["VAEEncodeTiled"]()
            self.controlnetloader = NODE_CLASS_MAPPINGS["ControlNetLoader"]()
            self.controlnetloader_697 = self.controlnetloader.load_controlnet(
                control_net_name="control_v11f1e_sd15_tile.pth"
            )
            self.freeu_v2 = NODE_CLASS_MAPPINGS["FreeU_V2"]()
            self.perturbedattentionguidance = NODE_CLASS_MAPPINGS["PerturbedAttentionGuidance"]()
            self.automatic_cfg = NODE_CLASS_MAPPINGS["Automatic CFG"]()
            self.tileddiffusion = NODE_CLASS_MAPPINGS["TiledDiffusion"]()
            self.controlnetapplyadvanced = NODE_CLASS_MAPPINGS["ControlNetApplyAdvanced"]()
            self.samplercustom = NODE_CLASS_MAPPINGS["SamplerCustom"]()
            self.vaedecodetiled = NODE_CLASS_MAPPINGS["VAEDecodeTiled"]()
            self.imagescale = NODE_CLASS_MAPPINGS["ImageScale"]()

            self.checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
            self.checkpointloadersimple_685 = self.checkpointloadersimple.load_checkpoint(
                ckpt_name="juggernaut_reborn.safetensors"
            )
            loraloader_687 = self.loraloader.load_lora(
                lora_name="v15/more_details.safetensors",
                strength_model=0.2,
                strength_clip=0.2,
                model=get_value_at_index(self.checkpointloadersimple_685, 0),
                clip=get_value_at_index(self.checkpointloadersimple_685, 1),
            )

            self.loraloader_686 = self.loraloader.load_lora(
                lora_name="v15/SDXLrender_v2.0.safetensors",
                strength_model=0.1,
                strength_clip=0.2,
                model=get_value_at_index(loraloader_687, 0),
                clip=get_value_at_index(loraloader_687, 1),
            )
            self.freeu_v2_691 = self.freeu_v2.patch(
                b1=0.9,
                b2=1.08,
                s1=0.9500000000000001,
                s2=0.8,
                model=get_value_at_index(self.loraloader_686, 0),
            )
            self.perturbedattentionguidance_675 = self.perturbedattentionguidance.patch(
                scale=1, model=get_value_at_index(self.freeu_v2_691, 0)
            )
            self.automatic_cfg_688 = self.automatic_cfg.patch(
                hard_mode=True,
                boost=True,
                model=get_value_at_index(self.perturbedattentionguidance_675, 0),
            )
            self.tileddiffusion_698 = self.tileddiffusion.apply(
                method="MultiDiffusion",
                tile_width=1024,
                tile_height=1024,
                tile_overlap=128,
                tile_batch_size=4,
                model=get_value_at_index(self.automatic_cfg_688, 0),
            )
            self.cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
            self.cliptextencode_676 = self.cliptextencode.encode(
                text="(worst quality, low quality, normal quality:1.5)",
                clip=get_value_at_index(self.loraloader_686, 1),
            )
            self.cliptextencode_684 = self.cliptextencode.encode(
                text="masterpiece, best quality, highres",
                clip=get_value_at_index(self.loraloader_686, 1),
            )

            self.alignyourstepsscheduler = NODE_CLASS_MAPPINGS["AlignYourStepsScheduler"]()

            ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
            self.ksamplerselect_678 = ksamplerselect.get_sampler(sampler_name="dpmpp_3m_sde_gpu")

            self.upscalemodelloader = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
            self.upscalemodelloader_690 = self.upscalemodelloader.load_model(
                model_name="RealESRGAN_x2plus.pth"
            )
            self.imagescaleby = NODE_CLASS_MAPPINGS["ImageScaleBy"]()

    def forward(self, input_image):
        with torch.inference_mode():
            # upscale use esrgan x2
            imageupscalewithmodel_695 = self.imageupscalewithmodel.upscale(
                upscale_model=get_value_at_index(self.upscalemodelloader_690, 0),
                image=get_value_at_index(input_image, 0),
            )
            
            imagescaleby_694 = self.imagescaleby.upscale(
                upscale_method="lanczos",
                scale_by=0.6000000000000001,
                image=get_value_at_index(imageupscalewithmodel_695, 0),
            )
            # encode as latent
            vaeencodetiled_693 = self.vaeencodetiled.encode(
                tile_size=1024,
                overlap=64,
                temporal_size=64,
                temporal_overlap=8,
                pixels=get_value_at_index(imagescaleby_694, 0),
                vae=get_value_at_index(self.checkpointloadersimple_685, 2),
            )

            alignyourstepsscheduler = NODE_CLASS_MAPPINGS["AlignYourStepsScheduler"]()
            alignyourstepsscheduler_677 = alignyourstepsscheduler.get_sigmas(
                model_type="SD1", steps=30, denoise=0.30000000000000004
            )
            controlnetapplyadvanced_699 = self.controlnetapplyadvanced.apply_controlnet(
                strength=1,
                start_percent=0.1,
                end_percent=1,
                positive=get_value_at_index(self.cliptextencode_684, 0),
                negative=get_value_at_index(self.cliptextencode_676, 0),
                control_net=get_value_at_index(self.controlnetloader_697, 0),
                image=get_value_at_index(input_image, 0),
            )

            samplercustom_689 = self.samplercustom.sample(
                add_noise=True,
                noise_seed=random.randint(1, 2**64),
                cfg=8,
                model=get_value_at_index(self.tileddiffusion_698, 0),
                positive=get_value_at_index(controlnetapplyadvanced_699, 0),
                negative=get_value_at_index(controlnetapplyadvanced_699, 1),
                sampler=get_value_at_index(self.ksamplerselect_678, 0),
                sigmas=get_value_at_index(alignyourstepsscheduler_677, 0),
                latent_image=get_value_at_index(vaeencodetiled_693, 0),
            )

            vaedecodetiled_679 = self.vaedecodetiled.decode(
                tile_size=1024,
                overlap=64,
                temporal_size=64,
                temporal_overlap=8,
                samples=get_value_at_index(samplercustom_689, 0),
                vae=get_value_at_index(self.checkpointloadersimple_685, 2),
            )
        # use original struct return
        return vaedecodetiled_679


lego_version = '_refiner_v1'
save_dir = '/home/dell/study/test_comfy/img'

import glob
from itertools import product
lego_version = 'tryon_v4'

base_dir = '/home/dell/study/test_comfy/img/'
record_log = f'/home/dell/study/test_comfy/1_{lego_version}_1_a600.txt'

cloth_ = base_dir + 'clothe/*.jpg'
human_ = base_dir + 'motel/'
clothes = glob.glob(cloth_)
humans = glob.glob(human_ + '*')
humans = [i for i in humans  if 'mask' not in i]
human_dc = {int(i.split('/')[-1].split('.')[0]) : i for i in humans }
human_dc = {k:v for k,v in human_dc.items() if k < 3}

human_masks = [i for i in glob.glob(human_ + '*')  if 'mask' in i]
human_mask_dc = {int(i.split('/')[-1].split('_mask.')[0]) : i for i in human_masks }

human_clothe_pairs = [
    {
        'human_id': human_id,
        'human_path': human_path,
        'human_mask_path': human_mask_dc[human_id],
        'cloth_path': cloth_path
    }
    for human_id, human_path in human_dc.items()
    for cloth_path in clothes
]

save_dir = f'/home/dell/study/test_comfy/img/{lego_version}/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


print('human_clothe_pairs', len(human_clothe_pairs))

def main():
    import_custom_nodes()
    # initiate an refiner model
    refiner = Refinerv1()
    clipvisionloader = NODE_CLASS_MAPPINGS["CLIPVisionLoader"]()
    clipvisionloader_329 = clipvisionloader.load_clip(
        clip_name="sigclip_vision_patch14_384.safetensors"
    )

    loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
    imageresizekj = NODE_CLASS_MAPPINGS["ImageResizeKJ"]()
    loadimagemask = NODE_CLASS_MAPPINGS["LoadImageMask"]()
    resizemask = NODE_CLASS_MAPPINGS["ResizeMask"]()
    growmaskwithblur = NODE_CLASS_MAPPINGS["GrowMaskWithBlur"]()
    layerutility_imagescalebyaspectratio_v2 = NODE_CLASS_MAPPINGS[
        "LayerUtility: ImageScaleByAspectRatio V2"
    ]()
    layerutility_imagemaskscaleas = NODE_CLASS_MAPPINGS[
        "LayerUtility: ImageMaskScaleAs"
    ]()
    layermask_loadbirefnetmodelv2 = NODE_CLASS_MAPPINGS[
        "LayerMask: LoadBiRefNetModelV2"
    ]()
    layermask_loadbirefnetmodelv2_272 = (
        layermask_loadbirefnetmodelv2.load_birefnet_model(version="RMBG-2.0")
    )
    layermask_birefnetultrav2 = NODE_CLASS_MAPPINGS["LayerMask: BiRefNetUltraV2"]()
    layerutility_imageremovealpha = NODE_CLASS_MAPPINGS[
        "LayerUtility: ImageRemoveAlpha"
    ]()
    clipvisionencode = NODE_CLASS_MAPPINGS["CLIPVisionEncode"]()
    dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
    dualcliploader_322 = dualcliploader.load_clip(
        clip_name1="t5xxl_fp8_e4m3fn.safetensors",
        clip_name2="clip_l.safetensors",
        type="flux",
        device="default",
    )
    vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
    vaeloader_328 = vaeloader.load_vae(vae_name="flux-vae-bf16.safetensors")
    easy_imageconcat = NODE_CLASS_MAPPINGS["easy imageConcat"]()
    solidmask = NODE_CLASS_MAPPINGS["SolidMask"]()
    masktoimage = NODE_CLASS_MAPPINGS["MaskToImage"]()
    imagetomask = NODE_CLASS_MAPPINGS["ImageToMask"]()
    inpaintmodelconditioning = NODE_CLASS_MAPPINGS["InpaintModelConditioning"]()
    unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
    unetloader_326 = unetloader.load_unet(
        unet_name="flux1-fill-dev-fp8.safetensors", weight_dtype="fp8_e4m3fn_fast"
    )
    loraloadermodelonly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
    loraloadermodelonly_447 = loraloadermodelonly.load_lora_model_only(
        lora_name="FLUX.1-Turbo-Alpha.safetensors",
        strength_model=1,
        model=get_value_at_index(unetloader_326, 0),
    )
    #  ---------------------v15vrefiner
    checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
    checkpointloadersimple_685 = checkpointloadersimple.load_checkpoint(
        ckpt_name="juggernaut_reborn.safetensors"
    )
    loraloader = NODE_CLASS_MAPPINGS["LoraLoader"]()
    loraloader_687 = loraloader.load_lora(
        lora_name="v15/more_details.safetensors",
        strength_model=0.2,
        strength_clip=0.2,
        model=get_value_at_index(checkpointloadersimple_685, 0),
        clip=get_value_at_index(checkpointloadersimple_685, 1),
    )

    loraloader_686 = loraloader.load_lora(
        lora_name="v15/SDXLrender_v2.0.safetensors",
        strength_model=0.1,
        strength_clip=0.2,
        model=get_value_at_index(loraloader_687, 0),
        clip=get_value_at_index(loraloader_687, 1),
    )

    cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
    cliptextencode_676 = cliptextencode.encode(
        text="(worst quality, low quality, normal quality:1.5)",
        clip=get_value_at_index(loraloader_686, 1),
    )
    ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
    ksamplerselect_678 = ksamplerselect.get_sampler(sampler_name="dpmpp_3m_sde_gpu")

    cliptextencode_684 = cliptextencode.encode(
        text="masterpiece, best quality, highres",
        clip=get_value_at_index(loraloader_686, 1),
    )

    upscalemodelloader = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
    upscalemodelloader_690 = upscalemodelloader.load_model(
        model_name="RealESRGAN_x2plus.pth"
    )

    with torch.inference_mode(), tqdm.tqdm(len(human_clothe_pairs)) as bar:

        # start Batch----------
        print('>>>>>>>>>>>>>>> statr geternate')
        start_time = time.time()

        for inx, human_cloeth in enumerate(human_clothe_pairs):

            start_time = time.time()

            # loadimage_229 = loadimage.load_image(image="2.jpg")        
            # loadimagemask_432 = loadimagemask.load_image(image="2_mask.jpg", channel="red")
            # loadimage_228 = loadimage.load_image(image="cloth_02.png")
            human_id = human_cloeth['human_id']

            save_img_res = f'{save_dir}{human_id}_{inx}_res_{lego_version}.png'
            print('>> trying to generate : ', save_img_res)
            if os.path.exists(save_img_res):
                bar.update(1)
                continue

            human_img = Image.open(human_cloeth['human_path'])
            loadimage_229 = (pil2tensor(human_img), )  # B, H, W, C = image.shape  not enough values to unpack (expected 4, got 3)
            # loadimage_229 = loadimage.load_image(image="2.jpg")

            # loadimagemask_432 = loadimagemask.load_image(image="2_mask.jpg", channel="red")
            mask_img = Image.open(human_cloeth['human_mask_path'])
            loadimagemask_432 = (pilmask2tensor(mask_img),)

            cloth_img = Image.open(human_cloeth['cloth_path'])
            loadimage_228 = (pil2tensor(cloth_img), )
            # loadimage_228 = loadimage.load_image(image="1 (1).jpg")

            imageresizekj_398 = imageresizekj.resize(
                width=1024,
                height=1024,
                upscale_method="nearest-exact",
                keep_proportion=True,
                divisible_by=2,
                crop="disabled",
                image=get_value_at_index(loadimage_229, 0),
            )

            resizemask_433 = resizemask.resize(
                width=get_value_at_index(imageresizekj_398, 1),
                height=get_value_at_index(imageresizekj_398, 2),
                keep_proportions=False,
                upscale_method="nearest-exact",
                crop="disabled",
                mask=get_value_at_index(loadimagemask_432, 0),
            )
            
            growmaskwithblur_337 = growmaskwithblur.expand_mask(
                expand=15,
                incremental_expandrate=0,
                tapered_corners=True,
                flip_input=False,
                blur_radius=10,
                lerp_alpha=1,
                decay_factor=1,
                fill_holes=False,
                mask=get_value_at_index(resizemask_433, 0),
            )

            layerutility_imagescalebyaspectratio_v2_267 = (
                layerutility_imagescalebyaspectratio_v2.image_scale_by_aspect_ratio(
                    aspect_ratio="original",
                    proportional_width=1,
                    proportional_height=1,
                    fit="crop",
                    method="lanczos",
                    round_to_multiple="8",
                    scale_to_side="longest",
                    scale_to_length=1280,
                    background_color="#000000",
                    image=get_value_at_index(imageresizekj_398, 0),
                    mask=get_value_at_index(growmaskwithblur_337, 0),
                )
            )

            # loadimage_229 = loadimage.load_image(image="2.jpg")        
            # loadimagemask_432 = loadimagemask.load_image(image="2_mask.jpg", channel="red")
            # loadimage_228 = loadimage.load_image(image="cloth_02.png")

            layerutility_imagemaskscaleas_268 = (
                layerutility_imagemaskscaleas.image_mask_scale_as(
                    fit="letterbox",
                    method="lanczos",
                    scale_as=get_value_at_index(
                        layerutility_imagescalebyaspectratio_v2_267, 0
                    ),
                    image=get_value_at_index(loadimage_228, 0),
                )
            )


            
            layermask_birefnetultrav2_271 = layermask_birefnetultrav2.birefnet_ultra_v2(
                detail_method="VITMatte",
                detail_erode=4,
                detail_dilate=2,
                black_point=0.01,
                white_point=0.4,
                process_detail=True,
                device="cuda",
                max_megapixels=2,
                image=get_value_at_index(layerutility_imagemaskscaleas_268, 0),
                birefnet_model=get_value_at_index(layermask_loadbirefnetmodelv2_272, 0),
            )

            layerutility_imageremovealpha_273 = (
                layerutility_imageremovealpha.image_remove_alpha(
                    fill_background=True,
                    background_color="#000000",
                    RGBA_image=get_value_at_index(layermask_birefnetultrav2_271, 0),
                )
            )


            clipvisionencode_172 = clipvisionencode.encode(
                crop="center",
                clip_vision=get_value_at_index(clipvisionloader_329, 0),
                image=get_value_at_index(layerutility_imageremovealpha_273, 0),
            )


            cliptextencodeflux = NODE_CLASS_MAPPINGS["CLIPTextEncodeFlux"]()
            cliptextencodeflux_323 = cliptextencodeflux.encode(
                clip_l="",
                t5xxl="",
                guidance=30,
                clip=get_value_at_index(dualcliploader_322, 0),
            )

            cliptextencodeflux_325 = cliptextencodeflux.encode(
                clip_l="",
                t5xxl="",
                guidance=30,
                clip=get_value_at_index(dualcliploader_322, 0),
            )


            easy_imageconcat_275 = easy_imageconcat.concat(
                direction="right",
                match_image_size=False,
                image1=get_value_at_index(layerutility_imageremovealpha_273, 0),
                image2=get_value_at_index(layerutility_imagescalebyaspectratio_v2_267, 0),
            )


            solidmask_278 = solidmask.solid(
                value=0,
                width=get_value_at_index(layerutility_imagescalebyaspectratio_v2_267, 3),
                height=get_value_at_index(layerutility_imagescalebyaspectratio_v2_267, 4),
            )

            masktoimage_281 = masktoimage.mask_to_image(
                mask=get_value_at_index(solidmask_278, 0)
            )

            mask_threshold_region = NODE_CLASS_MAPPINGS["Mask Threshold Region"]()
            mask_threshold_region_597 = mask_threshold_region.threshold_region(
                black_threshold=75,
                white_threshold=101,
                masks=get_value_at_index(layerutility_imagescalebyaspectratio_v2_267, 1),
            )

            masktoimage_598 = masktoimage.mask_to_image(
                mask=get_value_at_index(mask_threshold_region_597, 0)
            )

            easy_imageconcat_280 = easy_imageconcat.concat(
                direction="right",
                match_image_size=False,
                image1=get_value_at_index(masktoimage_281, 0),
                image2=get_value_at_index(masktoimage_598, 0),
            )

            imagetomask_283 = imagetomask.image_to_mask(
                channel="red", image=get_value_at_index(easy_imageconcat_280, 0)
            )


            inpaintmodelconditioning_220 = inpaintmodelconditioning.encode(
                noise_mask=True,
                positive=get_value_at_index(cliptextencodeflux_323, 0),
                negative=get_value_at_index(cliptextencodeflux_325, 0),
                vae=get_value_at_index(vaeloader_328, 0),
                pixels=get_value_at_index(easy_imageconcat_275, 0),
                mask=get_value_at_index(imagetomask_283, 0),
            )


            stylemodelloader = NODE_CLASS_MAPPINGS["StyleModelLoader"]()
            stylemodelloader_330 = stylemodelloader.load_style_model(
                style_model_name="flux1-redux-dev.safetensors"
            )

            easy_prompt = NODE_CLASS_MAPPINGS["easy prompt"]()
            easy_prompt_430 = easy_prompt.doit(
                text="",
                prefix="Select the prefix add to the text",
                subject="👤Select the subject add to the text",
                action="🎬Select the action add to the text",
                clothes="👚Select the clothes add to the text",
                environment="☀️Select the illumination environment add to the text",
                background="🎞️Select the background add to the text",
                nsfw="🔞Select the nsfw add to the text",
            )



            alignyourstepsscheduler = NODE_CLASS_MAPPINGS["AlignYourStepsScheduler"]()
            alignyourstepsscheduler_677 = alignyourstepsscheduler.get_sigmas(
                model_type="SD1", steps=30, denoise=0.30000000000000004
            )

            
            layerutility_getimagesize = NODE_CLASS_MAPPINGS["LayerUtility: GetImageSize"]()
            layerutility_getimagesize_321 = layerutility_getimagesize.get_image_size(
                image=get_value_at_index(easy_imageconcat_275, 0)
            )

            differentialdiffusion = NODE_CLASS_MAPPINGS["DifferentialDiffusion"]()
            differentialdiffusion_327 = differentialdiffusion.apply(
                model=get_value_at_index(loraloadermodelonly_447, 0)
            )

            modelsamplingflux = NODE_CLASS_MAPPINGS["ModelSamplingFlux"]()
            modelsamplingflux_320 = modelsamplingflux.patch(
                max_shift=1.2000000000000002,
                base_shift=0.5,
                width=get_value_at_index(layerutility_getimagesize_321, 0),
                height=get_value_at_index(layerutility_getimagesize_321, 1),
                model=get_value_at_index(differentialdiffusion_327, 0),
            )

            stylemodelapply = NODE_CLASS_MAPPINGS["StyleModelApply"]()
            stylemodelapply_171 = stylemodelapply.apply_stylemodel(
                strength=1,
                strength_type="multiply",
                conditioning=get_value_at_index(inpaintmodelconditioning_220, 0),
                style_model=get_value_at_index(stylemodelloader_330, 0),
                clip_vision_output=get_value_at_index(clipvisionencode_172, 0),
            )

            fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
            fluxguidance_223 = fluxguidance.append(
                guidance=35, conditioning=get_value_at_index(stylemodelapply_171, 0)
            )

            ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
            ksampler_102 = ksampler.sample(
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

            vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
            vaedecode_106 = vaedecode.decode(
                samples=get_value_at_index(ksampler_102, 0),
                vae=get_value_at_index(vaeloader_328, 0),
            )

            easy_imagesplitgrid = NODE_CLASS_MAPPINGS["easy imageSplitGrid"]()
            easy_imagesplitgrid_294 = easy_imagesplitgrid.doit(
                row=1, column=2, images=get_value_at_index(vaedecode_106, 0)
            )

            easy_imagessplitimage = NODE_CLASS_MAPPINGS["easy imagesSplitImage"]()
            easy_imagessplitimage_299 = easy_imagessplitimage.split(
                images=get_value_at_index(easy_imagesplitgrid_294, 0)
            )

            growmask = NODE_CLASS_MAPPINGS["GrowMask"]()
            growmask_440 = growmask.expand_mask(
                expand=313,
                tapered_corners=True,
                mask=get_value_at_index(layerutility_imagescalebyaspectratio_v2_267, 1),
            )

            maskblur = NODE_CLASS_MAPPINGS["MaskBlur+"]()
            maskblur_441 = maskblur.execute(
                amount=13, device="auto", mask=get_value_at_index(growmask_440, 0)
            )

            imagecompositemasked = NODE_CLASS_MAPPINGS["ImageCompositeMasked"]()
            imagecompositemasked_439 = imagecompositemasked.composite(
                x=0,
                y=0,
                resize_source=False,
                destination=get_value_at_index(
                    layerutility_imagescalebyaspectratio_v2_267, 0
                ),
                source=get_value_at_index(easy_imagessplitimage_299, 1),
                mask=get_value_at_index(maskblur_441, 0),
            )

            imageupscalewithmodel = NODE_CLASS_MAPPINGS["ImageUpscaleWithModel"]()
            imageupscalewithmodel_695 = imageupscalewithmodel.upscale(
                upscale_model=get_value_at_index(upscalemodelloader_690, 0),
                image=get_value_at_index(imagecompositemasked_439, 0),
            )

            imagescaleby = NODE_CLASS_MAPPINGS["ImageScaleBy"]()
            imagescaleby_694 = imagescaleby.upscale(
                upscale_method="lanczos",
                scale_by=0.6000000000000001,
                image=get_value_at_index(imageupscalewithmodel_695, 0),
            )

            vaeencodetiled = NODE_CLASS_MAPPINGS["VAEEncodeTiled"]()
            vaeencodetiled_693 = vaeencodetiled.encode(
                tile_size=1024,
                overlap=64,
                temporal_size=64,
                temporal_overlap=8,
                pixels=get_value_at_index(imagescaleby_694, 0),
                vae=get_value_at_index(checkpointloadersimple_685, 2),
            )

            controlnetloader = NODE_CLASS_MAPPINGS["ControlNetLoader"]()
            controlnetloader_697 = controlnetloader.load_controlnet(
                control_net_name="control_v11f1e_sd15_tile.pth"
            )

            get_image_size = NODE_CLASS_MAPPINGS["Get Image Size"]()
            freeu_v2 = NODE_CLASS_MAPPINGS["FreeU_V2"]()
            perturbedattentionguidance = NODE_CLASS_MAPPINGS["PerturbedAttentionGuidance"]()
            automatic_cfg = NODE_CLASS_MAPPINGS["Automatic CFG"]()
            tileddiffusion = NODE_CLASS_MAPPINGS["TiledDiffusion"]()
            controlnetapplyadvanced = NODE_CLASS_MAPPINGS["ControlNetApplyAdvanced"]()
            samplercustom = NODE_CLASS_MAPPINGS["SamplerCustom"]()
            vaedecodetiled = NODE_CLASS_MAPPINGS["VAEDecodeTiled"]()
            imagescale = NODE_CLASS_MAPPINGS["ImageScale"]()
            saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()
            image_comparer_rgthree = NODE_CLASS_MAPPINGS["Image Comparer (rgthree)"]()

            for q in range(1):
                masktoimage_282 = masktoimage.mask_to_image(
                    mask=get_value_at_index(layerutility_imagescalebyaspectratio_v2_267, 1)
                )

                get_image_size_436 = get_image_size.get_size(
                    image=get_value_at_index(loadimage_229, 0)
                )

                growmask_444 = growmask.expand_mask(
                    expand=10,
                    tapered_corners=True,
                    mask=get_value_at_index(loadimagemask_432, 0),
                )

                maskblur_445 = maskblur.execute(
                    amount=30, device="auto", mask=get_value_at_index(growmask_444, 0)
                )
                if 0:
                    freeu_v2_691 = freeu_v2.patch(
                        b1=0.9,
                        b2=1.08,
                        s1=0.9500000000000001,
                        s2=0.8,
                        model=get_value_at_index(loraloader_686, 0),
                    )

                    perturbedattentionguidance_675 = perturbedattentionguidance.patch(
                        scale=1, model=get_value_at_index(freeu_v2_691, 0)
                    )

                    automatic_cfg_688 = automatic_cfg.patch(
                        hard_mode=True,
                        boost=True,
                        model=get_value_at_index(perturbedattentionguidance_675, 0),
                    )

                    tileddiffusion_698 = tileddiffusion.apply(
                        method="MultiDiffusion",
                        tile_width=1024,
                        tile_height=1024,
                        tile_overlap=128,
                        tile_batch_size=4,
                        model=get_value_at_index(automatic_cfg_688, 0),
                    )

                    controlnetapplyadvanced_699 = controlnetapplyadvanced.apply_controlnet(
                        strength=1,
                        start_percent=0.1,
                        end_percent=1,
                        positive=get_value_at_index(cliptextencode_684, 0),
                        negative=get_value_at_index(cliptextencode_676, 0),
                        control_net=get_value_at_index(controlnetloader_697, 0),
                        image=get_value_at_index(imagecompositemasked_439, 0),
                    )

                    samplercustom_689 = samplercustom.sample(
                        add_noise=True,
                        noise_seed=random.randint(1, 2**64),
                        cfg=8,
                        model=get_value_at_index(tileddiffusion_698, 0),
                        positive=get_value_at_index(controlnetapplyadvanced_699, 0),
                        negative=get_value_at_index(controlnetapplyadvanced_699, 1),
                        sampler=get_value_at_index(ksamplerselect_678, 0),
                        sigmas=get_value_at_index(alignyourstepsscheduler_677, 0),
                        latent_image=get_value_at_index(vaeencodetiled_693, 0),
                    )

                    vaedecodetiled_679 = vaedecodetiled.decode(
                        tile_size=1024,
                        overlap=64,
                        temporal_size=64,
                        temporal_overlap=8,
                        samples=get_value_at_index(samplercustom_689, 0),
                        vae=get_value_at_index(checkpointloadersimple_685, 2),
                    )
                
                
                refiner_res = refiner.forward(imagecompositemasked_439)
                vaedecodetiled_679 = refiner_res


                imagescale_528 = imagescale.upscale(
                    upscale_method="bicubic",
                    width=get_value_at_index(get_image_size_436, 0),
                    height=get_value_at_index(get_image_size_436, 1),
                    crop="disabled",
                    image=get_value_at_index(vaedecodetiled_679, 0),
                )

                imagecompositemasked_527 = imagecompositemasked.composite(
                    x=0,
                    y=0,
                    resize_source=False,
                    destination=get_value_at_index(loadimage_229, 0),
                    source=get_value_at_index(imagescale_528, 0),
                    mask=get_value_at_index(maskblur_445, 0),
                )

                masktoimage_585 = masktoimage.mask_to_image(
                    mask=get_value_at_index(loadimagemask_432, 0)
                )

                masktoimage_586 = masktoimage.mask_to_image(
                    mask=get_value_at_index(growmask_444, 0)
                )

                saveimage_588 = saveimage.save_images(
                    filename_prefix="tryon_v4",
                    images=get_value_at_index(imagecompositemasked_527, 0),
                )

                image_comparer_rgthree_745 = image_comparer_rgthree.compare_images(
                    image_a=get_value_at_index(imagecompositemasked_439, 0),
                    image_b=get_value_at_index(imagecompositemasked_527, 0),
                )

                image_comparer_rgthree_747 = image_comparer_rgthree.compare_images(
                    image_a=get_value_at_index(imagescaleby_694, 0),
                    image_b=get_value_at_index(vaedecodetiled_679, 0),
                )

                image_comparer_rgthree_748 = image_comparer_rgthree.compare_images(
                    image_a=get_value_at_index(vaedecodetiled_679, 0),
                    image_b=get_value_at_index(imagecompositemasked_527, 0),
                )
                human_img = tensor2pil(loadimage_229[0])
                cloth_rmbg = tensor2pil(layerutility_imageremovealpha_273[0])
                human_cloth_concated = tensor2pil(easy_imageconcat_275[0])
                generated_raw = tensor2pil(get_value_at_index(easy_imagessplitimage_299, 1))
                # repaint_area_mask = tensor2pil(masks_subtract_577[0])
                repaint_area_res = tensor2pil(imagescale_528[0])

                refiner_img = tensor2pil(vaedecodetiled_679[0])
                res_img = tensor2pil(imagecompositemasked_527[0])
                endtime = time.time()
                cost_time = f'{endtime - start_time:.2f}'
                debug_image_collection = [
                    human_img.resize(generated_raw.size), 
                    cloth_rmbg.resize(size=generated_raw.size),
                    draw_text(res_img, f"generated_res cost:{cost_time}"),
                        # draw_text(human_cloth_concated, "human_cloth_concated"),
                    draw_text(generated_raw, "generated_raw"),
                    draw_text(refiner_img, "refiner"),
                    res_img.resize(size = generated_raw.size)
                ]
                debug_img = make_image_grid(debug_image_collection, cols=3, rows=2).convert('RGB')
                # debug_img.save(f'{save_dir}/try_on_refiner_debug_{lego_version}.jpg')
                debug_img.save(f'{save_dir}{human_id}_{inx}_res_{lego_version}.png')
                # ref.write(f"id:{human_id} _{inx}_ {lego_version} cost: {cost_time} sec\n")
                bar.update(1)

                # debug_img.save(f'{save_dir}{human_id}_{inx}_debug_{lego_version}.jpg')
                # res_image.save(f'{save_dir}{human_id}_{inx}_res_{lego_version}.png')


if __name__ == "__main__":
    main()
    # import_custom_nodes()
    # loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
    # loadimage_228 = loadimage.load_image(image="cloth_02.png")
    # refiner = Refinerv1()
    # refiner_res = refiner.forward(loadimage_228)
    # debug_img = make_image_grid(
    #     [
    #         tensor2pil(loadimage_228[0]),
    #         tensor2pil(refiner_res[0]),
    #     ], cols=2, rows=1).convert('RGB')
    # debug_img.save(f'{save_dir}/1_refiner_debug_{lego_version}.jpg')

