import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import time
import numpy as np
import tqdm
from PIL import Image, ImageDraw, ImageFont
from diffusers.utils import make_image_grid

def pil2tensor(image):
    new_image = image.convert('RGB')
    img_array = np.array(new_image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array)[None]
    return img_tensor
def tensor2pil(image):
    return Image.fromarray((image[0].cpu().numpy() * 255).astype(np.uint8))

def pilmask2tensor(mask_img):
    mask_tensor = torch.from_numpy(np.array(mask_img.convert('L'))).float()  # 转换为float类型
    mask_tensor = mask_tensor / 255.0  # 归一化到 0-1 范围
    mask_tensor = mask_tensor.unsqueeze(0)
    return mask_tensor


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


import glob
from itertools import product
lego_version = 'tryon_v4'

base_dir = '/home/dell/study/test_comfy/'
save_dir = '/home/dell/study/test_comfy/img/'
record_log = f'/home/dell/study/test_comfy/1_{lego_version}_1_a600.txt'


cloth_ = base_dir + 'clothe/*.jpg'
human_ = base_dir + 'motel/'
clothes = glob.glob(cloth_)
humans = glob.glob(human_ + '*')
humans = [i for i in humans  if 'mask' not in i]
human_dc = {int(i.split('/')[-1].split('.')[0]) : i for i in humans }
human_dc = {k:v for k,v in human_dc.items() if k < 8}

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

def draw_text(image, text, position=(50, 50), font_size=45, color=(255, 255, 255)):  # 默认白色
    draw = ImageDraw.Draw(image)
    # 根据图像模式选择适当的颜色格式
    if image.mode == 'RGB':
        color = (255, 255, 255) if color == 255 else color  # RGB模式
    elif image.mode == 'RGBA':
        color = (255, 255, 255, 255) if color == 255 else color  # RGBA模式
    elif image.mode in ['L', '1']:
        color = 255 if isinstance(color, tuple) else color  # 灰度图模式
    
    font = ImageFont.load_default(size=font_size)
    
    # 绘制文本
    draw.text(position, text, font=font, fill=color)
    
    return image

# Save the data
def save_clip_data(data, path):
    data = data[0]
    torch.save({
        'tensor': data[0],  # Save the tensor
        'pooled_output': data[1]['pooled_output'],
        'guidance': data[1]['guidance']
    }, path)

# Load the data
def load_clip_data(path):
    data = torch.load(path)
    data = [data['tensor'], {
        'pooled_output': data['pooled_output'],
        'guidance': data['guidance']
    }]
    return [data]

cond1_save_path = '/home/dell/study/test_comfy/clip_data_cond1.pt'
cond2_save_path = '/home/dell/study/test_comfy/clip_data_cond2.pt'

def main():
    import_custom_nodes()
    # model loade ------
    clipvisionloader = NODE_CLASS_MAPPINGS["CLIPVisionLoader"]()
    clipvisionloader_329 = clipvisionloader.load_clip( clip_name="sigclip_vision_patch14_384.safetensors" )
    cond1_save_path = '/home/dell/study/test_comfy/clip_data_cond1.pt'
    cond2_save_path = '/home/dell/study/test_comfy/clip_data_cond2.pt'
    

    # Load later
    if os.path.exists(cond1_save_path):
        load_cond1 = load_clip_data(cond1_save_path)
        load_cond2 = load_clip_data(cond2_save_path)
    else:
        dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        dualcliploader_322 = dualcliploader.load_clip(
            clip_name1="t5xxl_fp16.safetensors", clip_name2="clip_l.safetensors",
            type="flux", device="default",
        )

        cliptextencodeflux = NODE_CLASS_MAPPINGS["CLIPTextEncodeFlux"]()
        cliptextencodeflux_323 = cliptextencodeflux.encode(
            clip_l="", t5xxl="", guidance=30,
            clip=get_value_at_index(dualcliploader_322, 0),
        )

        cliptextencodeflux_325 = cliptextencodeflux.encode(
            clip_l="", t5xxl="", guidance=30,
            clip=get_value_at_index(dualcliploader_322, 0),
        )
        save_clip_data(cliptextencodeflux_323[0], cond1_save_path)
        save_clip_data(cliptextencodeflux_325[0], cond2_save_path)
    cliptextencodeflux_323 = (load_cond1,)
    cliptextencodeflux_325 = (load_cond2,)

    vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
    vaeloader_328 = vaeloader.load_vae( vae_name="flux_vae/diffusion_pytorch_model.safetensors" )

    clipvisionencode = NODE_CLASS_MAPPINGS["CLIPVisionEncode"]()
    
    # flux fill need
    inpaintmodelconditioning = NODE_CLASS_MAPPINGS["InpaintModelConditioning"]()
    unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
    unetloader_326 = unetloader.load_unet(
        unet_name="flux1-fill-dev-fp8.safetensors", weight_dtype="fp8_e4m3fn_fast"
    )
    stylemodelloader = NODE_CLASS_MAPPINGS["StyleModelLoader"]()
    stylemodelloader_330 = stylemodelloader.load_style_model(
        style_model_name="flux1-redux-dev.safetensors"
    )
    loraloadermodelonly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
    loraloadermodelonly_447 = loraloadermodelonly.load_lora_model_only(
        lora_name="FLUX.1-Turbo-Alpha.safetensors",
        strength_model=1,
        model=get_value_at_index(unetloader_326, 0),
    )
    differentialdiffusion = NODE_CLASS_MAPPINGS["DifferentialDiffusion"]()
    differentialdiffusion_327 = differentialdiffusion.apply(
        model=get_value_at_index(loraloadermodelonly_447, 0)
    )
    pathchsageattentionkj = NODE_CLASS_MAPPINGS["PathchSageAttentionKJ"]()
    pathchsageattentionkj_449 = pathchsageattentionkj.patch(
        sage_attention="disabled",
        model=get_value_at_index(differentialdiffusion_327, 0),
    )
    
    stylemodelapply = NODE_CLASS_MAPPINGS["StyleModelApply"]()
    
    # sampleing 
    modelsamplingflux = NODE_CLASS_MAPPINGS["ModelSamplingFlux"]()   # 实时运行需要，hack 设置宽高之类的
    fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
    ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
    vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()

    # 图像处理工具节点~
    loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
    imageresizekj = NODE_CLASS_MAPPINGS["ImageResizeKJ"]()
    loadimagemask = NODE_CLASS_MAPPINGS["LoadImageMask"]()
    resizemask = NODE_CLASS_MAPPINGS["ResizeMask"]()
    growmaskwithblur = NODE_CLASS_MAPPINGS["GrowMaskWithBlur"]()
    layerutility_imagescalebyaspectratio_v2 = NODE_CLASS_MAPPINGS["LayerUtility: ImageScaleByAspectRatio V2"]()
    layerutility_imagemaskscaleas = NODE_CLASS_MAPPINGS[ "LayerUtility: ImageMaskScaleAs"]()
    layermask_loadbirefnetmodelv2 = NODE_CLASS_MAPPINGS["LayerMask: LoadBiRefNetModelV2"]()
    layermask_loadbirefnetmodelv2_272 = (
        layermask_loadbirefnetmodelv2.load_birefnet_model(version="RMBG-2.0")
    )
    layermask_birefnetultrav2 = NODE_CLASS_MAPPINGS["LayerMask: BiRefNetUltraV2"]()
    layerutility_imageremovealpha = NODE_CLASS_MAPPINGS["LayerUtility: ImageRemoveAlpha"]()
    easy_imageconcat = NODE_CLASS_MAPPINGS["easy imageConcat"]()
    solidmask = NODE_CLASS_MAPPINGS["SolidMask"]()
    masktoimage = NODE_CLASS_MAPPINGS["MaskToImage"]()
    imagetomask = NODE_CLASS_MAPPINGS["ImageToMask"]()
    layerutility_getimagesize = NODE_CLASS_MAPPINGS["LayerUtility: GetImageSize"]()
    easy_imagesplitgrid = NODE_CLASS_MAPPINGS["easy imageSplitGrid"]()
    easy_imagessplitimage = NODE_CLASS_MAPPINGS["easy imagesSplitImage"]()
    growmask = NODE_CLASS_MAPPINGS["GrowMask"]()
    maskblur = NODE_CLASS_MAPPINGS["MaskBlur+"]()
    imagecompositemasked = NODE_CLASS_MAPPINGS["ImageCompositeMasked"]()  # 组合图像
    masks_add = NODE_CLASS_MAPPINGS["Masks Add"]()
    masks_subtract = NODE_CLASS_MAPPINGS["Masks Subtract"]()

    get_image_size = NODE_CLASS_MAPPINGS["Get Image Size"]()
    imagescale = NODE_CLASS_MAPPINGS["ImageScale"]()
    image_comparer_rgthree = NODE_CLASS_MAPPINGS["Image Comparer (rgthree)"]()
    saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

    with torch.inference_mode() , open(record_log, 'w') as ref, tqdm.tqdm(len(human_clothe_pairs)) as bar:

        for inx, human_cloeth in enumerate(human_clothe_pairs):
            
            start_time = time.time()
            debug_image_collection = []

            human_id = human_cloeth['human_id']

            save_img_res = f'{save_dir}{human_id}_{inx}_res_{lego_version}.png'
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

            masktoimage = NODE_CLASS_MAPPINGS["MaskToImage"]()
            masktoimage_281 = masktoimage.mask_to_image(
                mask=get_value_at_index(solidmask_278, 0)
            )

            masktoimage_282 = masktoimage.mask_to_image(
                mask=get_value_at_index(layerutility_imagescalebyaspectratio_v2_267, 1)
            )

            easy_imageconcat_280 = easy_imageconcat.concat(
                direction="right",
                match_image_size=False,
                image1=get_value_at_index(masktoimage_281, 0),
                image2=get_value_at_index(masktoimage_282, 0),
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


            layerutility_getimagesize_321 = layerutility_getimagesize.get_image_size(
                image=get_value_at_index(easy_imageconcat_275, 0)
            )

            # 
            modelsamplingflux_320 = modelsamplingflux.patch(  
                max_shift=1.2000000000000002,
                base_shift=0.5,
                width=get_value_at_index(layerutility_getimagesize_321, 0),
                height=get_value_at_index(layerutility_getimagesize_321, 1),
                model=get_value_at_index(pathchsageattentionkj_449, 0),
            )

            
            stylemodelapply_171 = stylemodelapply.apply_stylemodel(
                strength=1,
                strength_type="multiply",
                conditioning=get_value_at_index(inpaintmodelconditioning_220, 0),
                style_model=get_value_at_index(stylemodelloader_330, 0),
                clip_vision_output=get_value_at_index(clipvisionencode_172, 0),
            )

            fluxguidance_223 = fluxguidance.append(
                guidance=30, conditioning=get_value_at_index(stylemodelapply_171, 0)
            )

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

            vaedecode_106 = vaedecode.decode(
                samples=get_value_at_index(ksampler_102, 0),
                vae=get_value_at_index(vaeloader_328, 0),
            )

            easy_imagesplitgrid_294 = easy_imagesplitgrid.doit(
                row=1, column=2, images=get_value_at_index(vaedecode_106, 0)
            )

            easy_imagessplitimage_299 = easy_imagessplitimage.split(
                images=get_value_at_index(easy_imagesplitgrid_294, 0)
            )

            growmask_440 = growmask.expand_mask(
                expand=313,
                tapered_corners=True,
                mask=get_value_at_index(layerutility_imagescalebyaspectratio_v2_267, 1),
            )

            maskblur_441 = maskblur.execute(
                amount=13, device="auto", mask=get_value_at_index(growmask_440, 0)
            )

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

            growmaskwithblur_569 = growmaskwithblur.expand_mask(
                expand=20,
                incremental_expandrate=0,
                tapered_corners=True,
                flip_input=False,
                blur_radius=15.600000000000001,
                lerp_alpha=1,
                decay_factor=1,
                fill_holes=False,
                mask=get_value_at_index(layerutility_imagescalebyaspectratio_v2_267, 1),
            )

            growmaskwithblur_584 = growmaskwithblur.expand_mask(
                expand=-10,
                incremental_expandrate=0,
                tapered_corners=True,
                flip_input=False,
                blur_radius=10,
                lerp_alpha=1,
                decay_factor=1,
                fill_holes=False,
                mask=get_value_at_index(layerutility_imagescalebyaspectratio_v2_267, 1),
            )

            masks_subtract_577 = masks_subtract.subtract_masks(
                masks_a=get_value_at_index(growmaskwithblur_569, 0),
                masks_b=get_value_at_index(growmaskwithblur_584, 0),
            )

            layerutility_imagescalebyaspectratio_v2_557 = (
                layerutility_imagescalebyaspectratio_v2.image_scale_by_aspect_ratio(
                    aspect_ratio="original",
                    proportional_width=1,
                    proportional_height=1,
                    fit="letterbox",
                    method="nearest",
                    round_to_multiple="None",
                    scale_to_side="longest",
                    scale_to_length=1280,
                    background_color="#000000",
                    image=get_value_at_index(imagecompositemasked_439, 0),
                    mask=get_value_at_index(masks_subtract_577, 0),
                )
            )

            inpaintmodelconditioning_560 = inpaintmodelconditioning.encode(
                noise_mask=False,
                positive=get_value_at_index(cliptextencodeflux_323, 0),
                negative=get_value_at_index(cliptextencodeflux_325, 0),
                vae=get_value_at_index(vaeloader_328, 0),
                pixels=get_value_at_index(layerutility_imagescalebyaspectratio_v2_557, 0),
                mask=get_value_at_index(layerutility_imagescalebyaspectratio_v2_557, 1),
            )
            # pure inf
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

            ksampler_559 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=8,
                cfg=1,
                sampler_name="euler",
                scheduler="beta",
                denoise=0.7000000000000001,
                model=get_value_at_index(pathchsageattentionkj_449, 0),
                positive=get_value_at_index(inpaintmodelconditioning_560, 0),
                negative=get_value_at_index(inpaintmodelconditioning_560, 1),
                latent_image=get_value_at_index(inpaintmodelconditioning_560, 2),
            )

            vaedecode_558 = vaedecode.decode(
                samples=get_value_at_index(ksampler_559, 0),
                vae=get_value_at_index(vaeloader_328, 0),
            )

            imagescale_528 = imagescale.upscale(
                upscale_method="bicubic",
                width=get_value_at_index(get_image_size_436, 0),
                height=get_value_at_index(get_image_size_436, 1),
                crop="disabled",
                image=get_value_at_index(vaedecode_558, 0),
            )

            imagecompositemasked_527 = imagecompositemasked.composite(
                x=0,
                y=0,
                resize_source=False,
                destination=get_value_at_index(loadimage_229, 0),
                source=get_value_at_index(imagescale_528, 0),
                mask=get_value_at_index(maskblur_445, 0),
            )

            image_comparer_rgthree_561 = image_comparer_rgthree.compare_images(
                image_a=get_value_at_index(vaedecode_558, 0),
                image_b=get_value_at_index(imagecompositemasked_439, 0),
            )

            masktoimage_572 = masktoimage.mask_to_image(
                mask=get_value_at_index(masks_subtract_577, 0)
            )

            masktoimage_574 = masktoimage.mask_to_image(
                mask=get_value_at_index(layerutility_imagescalebyaspectratio_v2_267, 1)
            )

            image_comparer_rgthree_575 = image_comparer_rgthree.compare_images(
                image_a=get_value_at_index(masktoimage_574, 0),
                image_b=get_value_at_index(masktoimage_572, 0),
            )

            masktoimage_585 = masktoimage.mask_to_image(
                mask=get_value_at_index(loadimagemask_432, 0)
            )

            masktoimage_586 = masktoimage.mask_to_image(
                mask=get_value_at_index(growmask_444, 0)
            )

            image_comparer_rgthree_587 = image_comparer_rgthree.compare_images(
                image_a=get_value_at_index(masktoimage_585, 0),
                image_b=get_value_at_index(masktoimage_586, 0),
            )

            # saveimage_588 = saveimage.save_images(
            #     filename_prefix="tryon_v4",
            #     images=get_value_at_index(imagecompositemasked_527, 0),
            # )
            endtime = time.time()
            res_image = tensor2pil(imagecompositemasked_527[0])

            human_resized = tensor2pil(layerutility_imagescalebyaspectratio_v2_267[0])
            mask_resized = tensor2pil(layerutility_imagescalebyaspectratio_v2_267[0])
            cloth_rmbg = tensor2pil(layerutility_imageremovealpha_273[0])
            human_cloth_concated = tensor2pil(easy_imageconcat_275[0])
            generated_raw = tensor2pil(get_value_at_index(easy_imagessplitimage_299, 1))
            repaint_area_mask = tensor2pil(masks_subtract_577[0])
            repaint_area_res = tensor2pil(imagescale_528[0])
            cost_time = f'{endtime - start_time:.2f}'
            debug_image_collection = [
                human_img, 
                draw_text(cloth_rmbg.resize(size=human_img.size),
                        human_cloeth['human_path'].split('/')[-1] ),
                draw_text(res_image, f"generated_res cost:{cost_time}"),
                    # draw_text(human_cloth_concated, "human_cloth_concated"),
                draw_text(generated_raw, "generated_raw"),
                draw_text(repaint_area_mask, "repaint_area_mask"),
                draw_text(repaint_area_res, "repaint_area_res"),
            ]
            debug_img = make_image_grid(debug_image_collection, cols=3, rows=2).convert('RGB')
            debug_img.save(f'{save_dir}{human_id}_{inx}_debug_{lego_version}.jpg')
            res_image.save(f'{save_dir}{human_id}_{inx}_res_{lego_version}.png')
            ref.write(f"id:{human_id} _{inx}_ {lego_version} cost: {cost_time} sec\n")
            bar.update(1)
            
            # break

                
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time} seconds to run.")
        return result
    return wrapper


if __name__ == "__main__":
    main()
