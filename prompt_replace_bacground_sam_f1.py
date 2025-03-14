import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


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


def main():
    import_custom_nodes()
    with torch.inference_mode():
        checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
        checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
            ckpt_name="realisticVisionV60B1_v51HyperVAE.safetensors"
        )

        eg_zy_wbk = NODE_CLASS_MAPPINGS["EG_ZY_WBK"]()
        eg_zy_wbk_370 = eg_zy_wbk.convert_number_types(
            number1="detail, 4k, shopping mall"
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_6 = cliptextencode.encode(
            text=get_value_at_index(eg_zy_wbk_370, 2),
            clip=get_value_at_index(checkpointloadersimple_4, 1),
        )

        cliptextencode_7 = cliptextencode.encode(
            text="nsfw, owres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry,missing fingers,bad hands,missing arms, long neck, Humpbacked\n",
            clip=get_value_at_index(checkpointloadersimple_4, 1),
        )

        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_10 = loadimage.load_image(
            image="8792f460-860d-46f3-b4b7-fee154238f0a.jpg"
        )

        dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        dualcliploader_232 = dualcliploader.load_clip(
            clip_name1="clip_l.safetensors",
            clip_name2="t5xxl_fp8_e4m3fn.safetensors",
            type="flux",
            device="default",
        )

        cliptextencode_225 = cliptextencode.encode(
            text=get_value_at_index(eg_zy_wbk_370, 2),
            clip=get_value_at_index(dualcliploader_232, 0),
        )

        unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        unetloader_230 = unetloader.load_unet(
            unet_name="flux1-dev-fp8.safetensors", weight_dtype="default"
        )

        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_231 = vaeloader.load_vae(
            vae_name="flux_vae/diffusion_pytorch_model.safetensors"
        )

        cliptextencode_239 = cliptextencode.encode(
            text=get_value_at_index(eg_zy_wbk_370, 2),
            clip=get_value_at_index(dualcliploader_232, 0),
        )

        fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
        fluxguidance_229 = fluxguidance.append(
            guidance=15, conditioning=get_value_at_index(cliptextencode_239, 0)
        )

        layerutility_imagescalebyaspectratio_v2 = NODE_CLASS_MAPPINGS[
            "LayerUtility: ImageScaleByAspectRatio V2"
        ]()
        layerutility_imagescalebyaspectratio_v2_11 = (
            layerutility_imagescalebyaspectratio_v2.image_scale_by_aspect_ratio(
                aspect_ratio="original",
                proportional_width=1,
                proportional_height=1,
                fit="letterbox",
                method="lanczos",
                round_to_multiple="8",
                scale_to_side="width",
                scale_to_length=1024,
                background_color="#000000",
                image=get_value_at_index(loadimage_10, 0),
            )
        )

        layermask_segmentanythingultra_v2 = NODE_CLASS_MAPPINGS[
            "LayerMask: SegmentAnythingUltra V2"
        ]()
        layermask_segmentanythingultra_v2_12 = (
            layermask_segmentanythingultra_v2.segment_anything_ultra_v2(
                sam_model="sam_hq_vit_b (379MB)",
                grounding_dino_model="GroundingDINO_SwinT_OGC (694MB)",
                threshold=0.4,
                detail_method="VITMatte",
                detail_erode=6,
                detail_dilate=6,
                black_point=0.15,
                white_point=0.99,
                process_detail=True,
                prompt="subject",
                device="cuda",
                max_megapixels=2,
                cache_model=False,
                image=get_value_at_index(layerutility_imagescalebyaspectratio_v2_11, 0),
            )
        )

        layermask_maskedgeultradetail_v2 = NODE_CLASS_MAPPINGS[
            "LayerMask: MaskEdgeUltraDetail V2"
        ]()
        layermask_maskedgeultradetail_v2_13 = (
            layermask_maskedgeultradetail_v2.mask_edge_ultra_detail_v2(
                method="VITMatte",
                mask_grow=0,
                fix_gap=0,
                fix_threshold=0.75,
                edge_erode=6,
                edte_dilate=6,
                black_point=0.01,
                white_point=0.99,
                device="cuda",
                max_megapixels=2,
                image=get_value_at_index(layermask_segmentanythingultra_v2_12, 0),
                mask=get_value_at_index(layermask_segmentanythingultra_v2_12, 1),
            )
        )

        layerutility_imageremovealpha = NODE_CLASS_MAPPINGS[
            "LayerUtility: ImageRemoveAlpha"
        ]()
        layerutility_imageremovealpha_280 = (
            layerutility_imageremovealpha.image_remove_alpha(
                fill_background=True,
                background_color="#000000",
                RGBA_image=get_value_at_index(layermask_maskedgeultradetail_v2_13, 0),
            )
        )

        canny = NODE_CLASS_MAPPINGS["Canny"]()
        canny_278 = canny.detect_edge(
            low_threshold=0.1,
            high_threshold=0.4,
            image=get_value_at_index(layerutility_imageremovealpha_280, 0),
        )

        instructpixtopixconditioning = NODE_CLASS_MAPPINGS[
            "InstructPixToPixConditioning"
        ]()
        instructpixtopixconditioning_233 = instructpixtopixconditioning.encode(
            positive=get_value_at_index(fluxguidance_229, 0),
            negative=get_value_at_index(cliptextencode_225, 0),
            vae=get_value_at_index(vaeloader_231, 0),
            pixels=get_value_at_index(canny_278, 0),
        )

        loraloadermodelonly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
        loraloadermodelonly_234 = loraloadermodelonly.load_lora_model_only(
            lora_name="flux1-canny-dev-lora.safetensors",
            strength_model=1,
            model=get_value_at_index(unetloader_230, 0),
        )

        easy_globalseed = NODE_CLASS_MAPPINGS["easy globalSeed"]()
        easy_globalseed_241 = easy_globalseed.doit(
            value=246828200854702,
            mode=False,
            action="randomize",
            last_seed=504564500719328,
        )

        loraloadermodelonly_384 = loraloadermodelonly.load_lora_model_only(
            lora_name="FLUX.1-Turbo-Alpha.safetensors",
            strength_model=1,
            model=get_value_at_index(loraloadermodelonly_234, 0),
        )

        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        ksampler_224 = ksampler.sample(
            seed=random.randint(1, 2**64),
            steps=8,
            cfg=1,
            sampler_name="euler",
            scheduler="normal",
            denoise=1,
            model=get_value_at_index(loraloadermodelonly_384, 0),
            positive=get_value_at_index(instructpixtopixconditioning_233, 0),
            negative=get_value_at_index(instructpixtopixconditioning_233, 1),
            latent_image=get_value_at_index(instructpixtopixconditioning_233, 2),
        )

        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        vaedecode_226 = vaedecode.decode(
            samples=get_value_at_index(ksampler_224, 0),
            vae=get_value_at_index(vaeloader_231, 0),
        )

        imagecompositemasked = NODE_CLASS_MAPPINGS["ImageCompositeMasked"]()
        imagecompositemasked_51 = imagecompositemasked.composite(
            x=0,
            y=0,
            resize_source=False,
            destination=get_value_at_index(vaedecode_226, 0),
            source=get_value_at_index(layerutility_imageremovealpha_280, 0),
            mask=get_value_at_index(layermask_maskedgeultradetail_v2_13, 1),
        )

        vaeencode = NODE_CLASS_MAPPINGS["VAEEncode"]()
        vaeencode_407 = vaeencode.encode(
            pixels=get_value_at_index(imagecompositemasked_51, 0),
            vae=get_value_at_index(checkpointloadersimple_4, 2),
        )

        easy_ipadapterapply = NODE_CLASS_MAPPINGS["easy ipadapterApply"]()
        detailtransfer = NODE_CLASS_MAPPINGS["DetailTransfer"]()
        splitimagewithalpha = NODE_CLASS_MAPPINGS["SplitImageWithAlpha"]()
        imageinvert = NODE_CLASS_MAPPINGS["ImageInvert"]()
        imagegaussianblur = NODE_CLASS_MAPPINGS["ImageGaussianBlur"]()
        image_blending_mode = NODE_CLASS_MAPPINGS["Image Blending Mode"]()
        impactgaussianblurmask = NODE_CLASS_MAPPINGS["ImpactGaussianBlurMask"]()
        growmask = NODE_CLASS_MAPPINGS["GrowMask"]()
        masktoimage = NODE_CLASS_MAPPINGS["MaskToImage"]()
        image_blend_by_mask = NODE_CLASS_MAPPINGS["Image Blend by Mask"]()
        layercolor_levels = NODE_CLASS_MAPPINGS["LayerColor: Levels"]()
        layercolor_colorbalance = NODE_CLASS_MAPPINGS["LayerColor: ColorBalance"]()
        layercolor_autoadjustv2 = NODE_CLASS_MAPPINGS["LayerColor: AutoAdjustV2"]()
        image_comparer_rgthree = NODE_CLASS_MAPPINGS["Image Comparer (rgthree)"]()
        easy_imagesize = NODE_CLASS_MAPPINGS["easy imageSize"]()
        previewmask_ = NODE_CLASS_MAPPINGS["PreviewMask_"]()

        for q in range(1):
            easy_ipadapterapply_17 = easy_ipadapterapply.apply(
                preset="PLUS (high strength)",
                lora_strength=0.6,
                provider="CPU",
                weight=1,
                weight_faceidv2=1,
                start_at=0,
                end_at=1,
                cache_mode="all",
                use_tiled=False,
                model=get_value_at_index(checkpointloadersimple_4, 0),
                image=get_value_at_index(layerutility_imagescalebyaspectratio_v2_11, 0),
                attn_mask=get_value_at_index(layermask_maskedgeultradetail_v2_13, 1),
            )

            ksampler_3 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=6,
                cfg=2,
                sampler_name="euler",
                scheduler="sgm_uniform",
                denoise=0.15,
                model=get_value_at_index(easy_ipadapterapply_17, 0),
                positive=get_value_at_index(cliptextencode_6, 0),
                negative=get_value_at_index(cliptextencode_7, 0),
                latent_image=get_value_at_index(vaeencode_407, 0),
            )

            vaedecode_405 = vaedecode.decode(
                samples=get_value_at_index(ksampler_3, 0),
                vae=get_value_at_index(checkpointloadersimple_4, 2),
            )

            detailtransfer_27 = detailtransfer.process(
                mode="add",
                blur_sigma=1,
                blend_factor=1,
                target=get_value_at_index(vaedecode_405, 0),
                source=get_value_at_index(layerutility_imageremovealpha_280, 0),
                mask=get_value_at_index(layermask_maskedgeultradetail_v2_13, 1),
            )

            splitimagewithalpha_136 = splitimagewithalpha.split_image_with_alpha(
                image=get_value_at_index(layerutility_imagescalebyaspectratio_v2_11, 0)
            )

            imageinvert_137 = imageinvert.invert(
                image=get_value_at_index(splitimagewithalpha_136, 0)
            )

            imagegaussianblur_138 = imagegaussianblur.image_gaussian_blur(
                radius=50, images=get_value_at_index(splitimagewithalpha_136, 0)
            )

            image_blending_mode_139 = image_blending_mode.image_blending_mode(
                mode="add",
                blend_percentage=0.5,
                image_a=get_value_at_index(imageinvert_137, 0),
                image_b=get_value_at_index(imagegaussianblur_138, 0),
            )

            imageinvert_140 = imageinvert.invert(
                image=get_value_at_index(image_blending_mode_139, 0)
            )

            image_blending_mode_141 = image_blending_mode.image_blending_mode(
                mode="add",
                blend_percentage=1,
                image_a=get_value_at_index(imagegaussianblur_138, 0),
                image_b=get_value_at_index(imageinvert_140, 0),
            )

            splitimagewithalpha_144 = splitimagewithalpha.split_image_with_alpha(
                image=get_value_at_index(detailtransfer_27, 0)
            )

            imagegaussianblur_146 = imagegaussianblur.image_gaussian_blur(
                radius=40, images=get_value_at_index(splitimagewithalpha_144, 0)
            )

            imageinvert_145 = imageinvert.invert(
                image=get_value_at_index(splitimagewithalpha_144, 0)
            )

            image_blending_mode_147 = image_blending_mode.image_blending_mode(
                mode="add",
                blend_percentage=0.5,
                image_a=get_value_at_index(imageinvert_145, 0),
                image_b=get_value_at_index(imagegaussianblur_146, 0),
            )

            imageinvert_148 = imageinvert.invert(
                image=get_value_at_index(image_blending_mode_147, 0)
            )

            image_blending_mode_149 = image_blending_mode.image_blending_mode(
                mode="add",
                blend_percentage=1,
                image_a=get_value_at_index(imagegaussianblur_146, 0),
                image_b=get_value_at_index(imageinvert_148, 0),
            )

            impactgaussianblurmask_159 = impactgaussianblurmask.doit(
                kernel_size=40,
                sigma=16.7,
                mask=get_value_at_index(layermask_maskedgeultradetail_v2_13, 1),
            )

            growmask_161 = growmask.expand_mask(
                expand=-10,
                tapered_corners=True,
                mask=get_value_at_index(impactgaussianblurmask_159, 0),
            )

            masktoimage_160 = masktoimage.mask_to_image(
                mask=get_value_at_index(growmask_161, 0)
            )

            image_blend_by_mask_151 = image_blend_by_mask.image_blend_mask(
                blend_percentage=1,
                image_a=get_value_at_index(image_blending_mode_149, 0),
                image_b=get_value_at_index(image_blending_mode_141, 0),
                mask=get_value_at_index(masktoimage_160, 0),
            )

            image_blending_mode_153 = image_blending_mode.image_blending_mode(
                mode="add",
                blend_percentage=0.66,
                image_a=get_value_at_index(imagegaussianblur_146, 0),
                image_b=get_value_at_index(image_blend_by_mask_151, 0),
            )

            layercolor_levels_347 = layercolor_levels.levels(
                channel="RGB",
                black_point=81,
                white_point=164,
                gray_point=1,
                output_black_point=0,
                output_white_point=255,
                image=get_value_at_index(image_blending_mode_153, 0),
            )

            layercolor_colorbalance_368 = layercolor_colorbalance.color_balance(
                cyan_red=0,
                magenta_green=0,
                yellow_blue=0,
                image=get_value_at_index(layercolor_levels_347, 0),
            )

            layercolor_autoadjustv2_285 = layercolor_autoadjustv2.auto_adjust_v2(
                strength=100,
                brightness=5,
                contrast=-2,
                saturation=4,
                red=0,
                green=0,
                blue=8,
                mode="RGB",
                image=get_value_at_index(layercolor_colorbalance_368, 0),
            )

            image_comparer_rgthree_215 = image_comparer_rgthree.compare_images(
                image_a=get_value_at_index(detailtransfer_27, 0),
                image_b=get_value_at_index(layercolor_autoadjustv2_285, 0),
            )

            easy_imagesize_258 = easy_imagesize.image_width_height(
                image=get_value_at_index(layermask_maskedgeultradetail_v2_13, 0)
            )

            easy_imagesize_259 = easy_imagesize.image_width_height(
                image=get_value_at_index(canny_278, 0)
            )

            previewmask__401 = previewmask_.run(
                mask=get_value_at_index(layermask_maskedgeultradetail_v2_13, 1)
            )


if __name__ == "__main__":
    main()
