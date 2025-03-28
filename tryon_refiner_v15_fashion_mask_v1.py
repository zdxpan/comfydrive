import os
import time
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import tqdm, time, os
from PIL import Image, ImageChops, ImageDraw
from src.util import (
    tensor2pil, pil2tensor, pilmask2tensor, make_image_grid, draw_text, mask2image,
    expand_face_box,get_value_at_index,MaskAdd,
)


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
# from src.sagspeed import zdxApplySageAtt
from src.mask_seg_det import  HumanSegmentParts, FashionSegDetect ,HumanFashionMaskDetailer
from src.human_mask_detail import HumanMaskSegDetTool
human_mask_detect_and_expand = HumanMaskSegDetTool.human_mask_detect_and_expand
human_mask_detect_and_expand_with_setting = HumanMaskSegDetTool.human_mask_detect_and_expand_with_setting

from src.refiner_v1  import Refinerv1

# class Refinerv1():
#     pass


import glob
from itertools import product


lego_version = 'tryon_fashion_mask_v2'
base_dir = '/home/dell/study/test_comfy/img/'
record_log = f'/home/dell/study/test_comfy/1_{lego_version}_1_a600.txt'

cloth_ = base_dir + 'tryon_no_mask/cloth_*'
human_ = base_dir + 'tryon_no_mask/image_*'
clothes = glob.glob(cloth_)
humans = glob.glob(human_)
human_dc = {int(i.split('image_')[-1].split('.')[0]): i for i in humans }
cloth_dc = {int(i.split('cloth_')[-1].split('.')[0]): i for i in clothes }
if 0:
    human_clothe_pairs = [
        {
            'human_id': human_id,
            'human_path': human_path,
            'cloth_path': cloth_path
        }
        for human_id, human_path in enumerate(humans)
        for cloth_path in clothes
    ]
human_clothe_pairs = [
    {
        'human_id': human_id,
        'human_path': human_path,
        'cloth_path': cloth_dc[human_id]
    }
    for human_id, human_path in human_dc.items()
]

mask_pil_im_ = '/home/dell/study/test_comfy/img/human_mask/mask_06.png'

print('>>>>>>>>_humans_cunt:', len(human_clothe_pairs))

save_dir = f'/home/dell/study/test_comfy/img/{lego_version}/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def main():
    import_custom_nodes()

    fashion_cls_model = '/home/dell/study/comfyui/models/yolo/deepfashion2_yolov8s-seg.pt'
    human_fashion_mask_model = HumanFashionMaskDetailer()
    fashion_detect_model =  FashionSegDetect(fashion_cls_model)
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
    # ----------- flux fill redux related-- 
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
    # --image and mask process
    easy_imageconcat = NODE_CLASS_MAPPINGS["easy imageConcat"]()
    solidmask = NODE_CLASS_MAPPINGS["SolidMask"]()
    masktoimage = NODE_CLASS_MAPPINGS["MaskToImage"]()
    imagetomask = NODE_CLASS_MAPPINGS["ImageToMask"]()
    growmask = NODE_CLASS_MAPPINGS["GrowMask"]()
    get_image_size = NODE_CLASS_MAPPINGS["Get Image Size"]()
    imagescale = NODE_CLASS_MAPPINGS["ImageScale"]() 


    with torch.inference_mode(), tqdm.tqdm(len(human_clothe_pairs)) as bar:

        # start Batch----------
        print('>>>>>>>>>>>>>>> statr geternate')
        start_time = time.time()

        for inx, human_cloeth in enumerate(human_clothe_pairs):

            start_time = time.time()

            # loadimage_229 = loadimage.load_image(image="2.jpg")        
            # loadimagemask_432 = loadimagemask.load_image(image="2_mask.jpg", channel="red")
            # loadimage_228_cloth = loadimage.load_image(image="cloth_02.png")
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
            
            if 'human_mask_path' not in human_cloeth:
                mask_img = None
                loadimagemask_432 = None
                # get human segment mask , better get the down_parts mask
                # human_mask_optimazed = human_mask_detect_and_expand(loadimage_229[0], human_fashion_mask_model) # [1,2048,1536]
                # loadimagemask_432 = human_mask_optimazed
            else:
                # test load a musk see what`s  shape 
                mask_img = Image.open(human_cloeth['human_mask_path'])
                loadimagemask_432 = (pilmask2tensor(mask_img),)       #  shape be like  1,2560, 1920

            cloth_img = Image.open(human_cloeth['cloth_path'])
            loadimage_228_cloth = (pil2tensor(cloth_img), )

            imageresizekj_398 = imageresizekj.resize( # huaman 等效1K缩放~ 再处理~
                width=1024,
                height=1024,
                upscale_method="nearest-exact",
                keep_proportion=True,
                divisible_by=2,
                crop="disabled",
                image=get_value_at_index(loadimage_229, 0),  # human_image
            )

            # if detect clothes fashion style  and process the mask~  my router stetage
            # 按照我的思路，什么衣服，匹配什么amsk~ 来实现，也许效果有提升~
            if 1:
                # 针对下体的衣服 做grow_mask 然后换装
                clothe_resize1k = imageresizekj.resize(
                    width=1024, height=1024, upscale_method="nearest-exact",
                    keep_proportion=True, divisible_by=2, crop="disabled",
                    image=get_value_at_index(loadimage_228_cloth, 0),  # clothe
                )
                fashion_det_res = fashion_detect_model(tensor2pil(clothe_resize1k[0]))
                human_mask_result2 = human_fashion_mask_model(imageresizekj_398[0], extra_setting=fashion_det_res[1])
                loadimagemask_432 = (human_mask_result2[1], )
                # loadimagemask_432 = (human_mask_growed3[0], )
                

                # use Max low_cover_mask: test in v2 seemed bad
                down_expand = False
                for fashion_type_ in fashion_det_res[1]:
                    if 'down_' in fashion_type_:   #  down_short  down_long  down_longlong
                        down_expand = True
                        break
                if down_expand:
                    human_mask_result2_big_low_part = human_fashion_mask_model(imageresizekj_398[0], extra_setting={'low_cover_big'})
                    mask_img_pil = tensor2pil(human_mask_result2_big_low_part[1])  #  mode=L size=1242x204
                    box_mask = mask_img_pil.getbbox()

                    if box_mask:
                        box_width = box_mask[2] - box_mask[0]
                        expantd_pixel = int(box_width / 6)
                        # 性能最高~ 质量有问题~
                        # human_mask_growed3 = horizontal_expand(human_mask_result2_big_low_part[1], expantd_pixel)[0]  # 性能最高~
                        human_mask_growed3 = growmask.expand_mask(
                            expand=expantd_pixel, tapered_corners=True, mask=human_mask_result2_big_low_part[1],
                        )
                        human_mask_add = MaskAdd().add_masks(human_mask_growed3[0], human_mask_result2[1])[0]
                        loadimagemask_432 = (human_mask_add, )    # (torch.Size([1, 2560, 1920]), )
                        # debug 
                        mask_img_pil = tensor2pil(human_mask_add)  #  mode=L size=1242x204
                        mask_img_pil1 = tensor2pil(human_mask_result2[1])
                        make_image_grid(
                            [   tensor2pil(clothe_resize1k[0]),
                                mask_img_pil1, mask_img_pil, 
                            ], cols=3, rows=1
                        ).save(f'/home/dell/study/test_comfy/img/1_tryon_human_mask_1_judgeby_cloth_{human_id}.png')


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

            layerutility_imagescalebyaspectratio_v2_267 = im_mask_scalebyaspectratio_v2_267 = (
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
                    image=get_value_at_index(imageresizekj_398, 0),    # human_image_resize1k
                    mask=get_value_at_index(growmaskwithblur_337, 0),  # mask_resize_1k
                )
            )

            # loadimage_229 = loadimage.load_image(image="2.jpg")        
            # loadimagemask_432 = loadimagemask.load_image(image="2_mask.jpg", channel="red")
            # loadimage_228_cloth = loadimage.load_image(image="cloth_02.png")

            layerutility_imagemaskscaleas_268 = (
                layerutility_imagemaskscaleas.image_mask_scale_as(
                    fit="letterbox",
                    method="lanczos",
                    scale_as=get_value_at_index(
                        im_mask_scalebyaspectratio_v2_267, 0
                    ),
                    image=get_value_at_index(loadimage_228_cloth, 0),
                )
            )


            # remove cloth_background
            layermask_birefnetultrav2_271_cloth = layermask_birefnetultrav2.birefnet_ultra_v2(
                detail_method="VITMatte",
                detail_erode=4,
                detail_dilate=2,
                black_point=0.01,
                white_point=0.4,
                process_detail=True,
                device="cuda",
                max_megapixels=2,
                image=get_value_at_index(layerutility_imagemaskscaleas_268, 0), # cloth
                birefnet_model=get_value_at_index(layermask_loadbirefnetmodelv2_272, 0),
            )

            layerutility_imageremovealpha_273_cloth_rmbg = (
                layerutility_imageremovealpha.image_remove_alpha(
                    fill_background=True,
                    background_color="#000000",
                    RGBA_image=get_value_at_index(layermask_birefnetultrav2_271_cloth, 0),
                )
            )

            # --- redux get image_clip_feature 
            clipvisionencode_172 = clipvisionencode.encode(
                crop="center",
                clip_vision=get_value_at_index(clipvisionloader_329, 0),
                image=get_value_at_index(layerutility_imageremovealpha_273_cloth_rmbg, 0),
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
                image1=get_value_at_index(layerutility_imageremovealpha_273_cloth_rmbg, 0),
                image2=get_value_at_index(im_mask_scalebyaspectratio_v2_267, 0),
            )


            solidmask_278 = solidmask.solid(
                value=0,
                width=get_value_at_index(im_mask_scalebyaspectratio_v2_267, 3),
                height=get_value_at_index(im_mask_scalebyaspectratio_v2_267, 4),
            )

            masktoimage_281 = masktoimage.mask_to_image(
                mask=get_value_at_index(solidmask_278, 0)
            )

            mask_threshold_region = NODE_CLASS_MAPPINGS["Mask Threshold Region"]()
            mask_threshold_region_597 = mask_threshold_region.threshold_region(
                black_threshold=75,
                white_threshold=101,
                masks=get_value_at_index(im_mask_scalebyaspectratio_v2_267, 1),
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

            if 1:   # fill redux parts
                # pass

                stylemodelloader = NODE_CLASS_MAPPINGS["StyleModelLoader"]()
                stylemodelloader_330 = stylemodelloader.load_style_model(
                    style_model_name="flux1-redux-dev.safetensors"
                )

            
            layerutility_getimagesize = NODE_CLASS_MAPPINGS["LayerUtility: GetImageSize"]()
            layerutility_getimagesize_321 = layerutility_getimagesize.get_image_size(
                image=get_value_at_index(easy_imageconcat_275, 0)
            )

            if 1:   # fill redux parts
                # pass

                differentialdiffusion = NODE_CLASS_MAPPINGS["DifferentialDiffusion"]()
                differentialdiffusion_327 = differentialdiffusion.apply(
                    model=get_value_at_index(loraloadermodelonly_447, 0)
                )
                
                # applyf1sage_speed = zdxApplySageAtt()
                # applyf1sage_speed_752 = applyf1sage_speed.patch(
                #     use_SageAttention=True,
                #     model=get_value_at_index(differentialdiffusion_327, 0),
                # )
                applyf1sage_speed_752 = differentialdiffusion_327



            modelsamplingflux = NODE_CLASS_MAPPINGS["ModelSamplingFlux"]()
            modelsamplingflux_320 = modelsamplingflux.patch(
                max_shift=1.2000000000000002,
                base_shift=0.5,
                width=get_value_at_index(layerutility_getimagesize_321, 0),
                height=get_value_at_index(layerutility_getimagesize_321, 1),
                model=get_value_at_index(applyf1sage_speed_752, 0),
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

            growmask_440 = growmask.expand_mask(
                expand=313,
                tapered_corners=True,
                mask=get_value_at_index(im_mask_scalebyaspectratio_v2_267, 1),
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
                    im_mask_scalebyaspectratio_v2_267, 0
                ),
                source=get_value_at_index(easy_imagessplitimage_299, 1),
                mask=get_value_at_index(maskblur_441, 0),
            )

            for q in range(1):
                masktoimage_282 = masktoimage.mask_to_image(
                    mask=get_value_at_index(im_mask_scalebyaspectratio_v2_267, 1)
                )
                # paste to final size ~
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
                
                # refiner_res = refiner.forward(imagecompositemasked_439)
                # vaedecodetiled_679 = refiner_res
                vaedecodetiled_679 = imagecompositemasked_439



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

                human_imageresize15k = imageresizekj.resize( # huaman 等效1K缩放~ 再处理~
                    width=1024, height=1024, upscale_method="nearest-exact",
                    keep_proportion=True, divisible_by=2, crop="disabled",
                    image=get_value_at_index(loadimage_229, 0),  # human_image
                )

                human_img = tensor2pil(human_imageresize15k[0])
                human_mask = tensor2pil(loadimagemask_432[0])
                cloth_rmbg = tensor2pil(layerutility_imageremovealpha_273_cloth_rmbg[0])
                # human_cloth_concated = tensor2pil(easy_imageconcat_275[0])
                # generated_raw = tensor2pil(get_value_at_index(easy_imagessplitimage_299, 1))
                # repaint_area_mask = tensor2pil(masks_subtract_577[0])
                repaint_area_res = tensor2pil(imagescale_528[0])

                refiner_img = tensor2pil(vaedecodetiled_679[0])
                res_img = tensor2pil(imagecompositemasked_527[0])

                endtime = time.time()
                cost_time = f'{endtime - start_time:.2f}'
                debug_image_collection = [
                    human_img, 
                    cloth_rmbg.resize(size=human_img.size),
                    draw_text(res_img.resize(size=human_img.size), f"generated cost:{cost_time}"),
                    draw_text(human_mask.resize(size=human_img.size), "human_mask"),
                    # draw_text(refiner_img, "refiner"),
                    # res_img.resize(size = generated_raw.size)
                ]
                debug_img = make_image_grid(debug_image_collection, cols=4, rows=1).convert('RGB')
                # debug_img.save(f'{save_dir}/try_on_refiner_debug_{lego_version}.jpg')
                debug_img.save(f'{save_dir}{human_id}_{inx}_res_{lego_version}.png')
                # ref.write(f"id:{human_id} _{inx}_ {lego_version} cost: {cost_time} sec\n")
                bar.update(1)

                # debug_img.save(f'{save_dir}{human_id}_{inx}_debug_{lego_version}.jpg')
                # res_image.save(f'{save_dir}{human_id}_{inx}_res_{lego_version}.png')


if __name__ == "__main__":
    print('>> run main or  refiner?  commnet this line!')
    if 1:
        main()
    else:
        # test refiner V1
        import_custom_nodes()
        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_228_cloth = loadimage.load_image(image="cloth_02.png")
        refiner = Refinerv1()
        refiner_res = refiner.forward(loadimage_228_cloth)
        debug_img = make_image_grid(
            [
                tensor2pil(loadimage_228_cloth[0]),
                tensor2pil(refiner_res[0]),
            ], cols=2, rows=1).convert('RGB')
        debug_img.save(f'{save_dir}/1_refiner_debug_{lego_version}.jpg')

