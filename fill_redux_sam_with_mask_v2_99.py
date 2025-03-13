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
        clipvisionloader = NODE_CLASS_MAPPINGS["CLIPVisionLoader"]()
        clipvisionloader_329 = clipvisionloader.load_clip(
            clip_name="sigclip_vision_patch14_384.safetensors"
        )

        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_229 = loadimage.load_image(image="2.jpg")

        imageresizekj = NODE_CLASS_MAPPINGS["ImageResizeKJ"]()
        imageresizekj_398 = imageresizekj.resize(
            width=1024,
            height=1024,
            upscale_method="nearest-exact",
            keep_proportion=True,
            divisible_by=2,
            crop="disabled",
            image=get_value_at_index(loadimage_229, 0),
        )

        loadimagemask = NODE_CLASS_MAPPINGS["LoadImageMask"]()
        loadimagemask_432 = loadimagemask.load_image(image="2_mask.jpg", channel="red")

        resizemask = NODE_CLASS_MAPPINGS["ResizeMask"]()
        resizemask_433 = resizemask.resize(
            width=get_value_at_index(imageresizekj_398, 1),
            height=get_value_at_index(imageresizekj_398, 2),
            keep_proportions=False,
            upscale_method="nearest-exact",
            crop="disabled",
            mask=get_value_at_index(loadimagemask_432, 0),
        )

        growmaskwithblur = NODE_CLASS_MAPPINGS["GrowMaskWithBlur"]()
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

        layerutility_imagescalebyaspectratio_v2 = NODE_CLASS_MAPPINGS[
            "LayerUtility: ImageScaleByAspectRatio V2"
        ]()
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

        loadimage_228 = loadimage.load_image(image="1 (9).jpg")

        layerutility_imagemaskscaleas = NODE_CLASS_MAPPINGS[
            "LayerUtility: ImageMaskScaleAs"
        ]()
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

        layermask_loadbirefnetmodelv2 = NODE_CLASS_MAPPINGS[
            "LayerMask: LoadBiRefNetModelV2"
        ]()
        layermask_loadbirefnetmodelv2_272 = (
            layermask_loadbirefnetmodelv2.load_birefnet_model(version="RMBG-2.0")
        )

        layermask_birefnetultrav2 = NODE_CLASS_MAPPINGS["LayerMask: BiRefNetUltraV2"]()
        layermask_birefnetultrav2_271 = layermask_birefnetultrav2.birefnet_ultra_v2(
            detail_method="VITMatte",
            detail_erode=4,
            detail_dilate=2,
            black_point=0.01,
            white_point=0.99,
            process_detail=False,
            device="cuda",
            max_megapixels=2,
            image=get_value_at_index(layerutility_imagemaskscaleas_268, 0),
            birefnet_model=get_value_at_index(layermask_loadbirefnetmodelv2_272, 0),
        )

        layerutility_imageremovealpha = NODE_CLASS_MAPPINGS[
            "LayerUtility: ImageRemoveAlpha"
        ]()
        layerutility_imageremovealpha_273 = (
            layerutility_imageremovealpha.image_remove_alpha(
                fill_background=True,
                background_color="#000000",
                RGBA_image=get_value_at_index(layermask_birefnetultrav2_271, 0),
            )
        )

        clipvisionencode = NODE_CLASS_MAPPINGS["CLIPVisionEncode"]()
        clipvisionencode_172 = clipvisionencode.encode(
            crop="center",
            clip_vision=get_value_at_index(clipvisionloader_329, 0),
            image=get_value_at_index(layerutility_imageremovealpha_273, 0),
        )

        dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        dualcliploader_322 = dualcliploader.load_clip(
            clip_name1="t5xxl_fp8_e4m3fn.safetensors",
            clip_name2="clip_l.safetensors",
            type="flux",
            device="default",
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

        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_328 = vaeloader.load_vae(
            vae_name="flux_vae/diffusion_pytorch_model.safetensors"
        )

        easy_imageconcat = NODE_CLASS_MAPPINGS["easy imageConcat"]()
        easy_imageconcat_275 = easy_imageconcat.concat(
            direction="right",
            match_image_size=False,
            image1=get_value_at_index(layerutility_imageremovealpha_273, 0),
            image2=get_value_at_index(layerutility_imagescalebyaspectratio_v2_267, 0),
        )

        solidmask = NODE_CLASS_MAPPINGS["SolidMask"]()
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

        imagetomask = NODE_CLASS_MAPPINGS["ImageToMask"]()
        imagetomask_283 = imagetomask.image_to_mask(
            channel="red", image=get_value_at_index(easy_imageconcat_280, 0)
        )

        inpaintmodelconditioning = NODE_CLASS_MAPPINGS["InpaintModelConditioning"]()
        inpaintmodelconditioning_220 = inpaintmodelconditioning.encode(
            noise_mask=True,
            positive=get_value_at_index(cliptextencodeflux_323, 0),
            negative=get_value_at_index(cliptextencodeflux_325, 0),
            vae=get_value_at_index(vaeloader_328, 0),
            pixels=get_value_at_index(easy_imageconcat_275, 0),
            mask=get_value_at_index(imagetomask_283, 0),
        )

        unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        unetloader_326 = unetloader.load_unet(
            unet_name="flux1-fill-dev-fp8.safetensors", weight_dtype="fp8_e4m3fn_fast"
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

        upscalemodelloader = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
        upscalemodelloader_434 = upscalemodelloader.load_model(
            model_name="4xRealWebPhoto_v4_dat2.pth"
        )

        loraloadermodelonly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
        loraloadermodelonly_447 = loraloadermodelonly.load_lora_model_only(
            lora_name="FLUX.1-Turbo-Alpha.safetensors",
            strength_model=1,
            model=get_value_at_index(unetloader_326, 0),
        )

        layerutility_getimagesize = NODE_CLASS_MAPPINGS["LayerUtility: GetImageSize"]()
        differentialdiffusion = NODE_CLASS_MAPPINGS["DifferentialDiffusion"]()
        modelsamplingflux = NODE_CLASS_MAPPINGS["ModelSamplingFlux"]()
        stylemodelapply = NODE_CLASS_MAPPINGS["StyleModelApply"]()
        fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        easy_imagesplitgrid = NODE_CLASS_MAPPINGS["easy imageSplitGrid"]()
        easy_imagessplitimage = NODE_CLASS_MAPPINGS["easy imagesSplitImage"]()
        easy_cleangpuused = NODE_CLASS_MAPPINGS["easy cleanGpuUsed"]()
        growmask = NODE_CLASS_MAPPINGS["GrowMask"]()
        maskblur = NODE_CLASS_MAPPINGS["MaskBlur+"]()
        imagecompositemasked = NODE_CLASS_MAPPINGS["ImageCompositeMasked"]()
        imageupscalewithmodel = NODE_CLASS_MAPPINGS["ImageUpscaleWithModel"]()
        get_image_size = NODE_CLASS_MAPPINGS["Get Image Size"]()
        imagescale = NODE_CLASS_MAPPINGS["ImageScale"]()
        saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

        for q in range(1):
            layerutility_getimagesize_321 = layerutility_getimagesize.get_image_size(
                image=get_value_at_index(easy_imageconcat_275, 0)
            )

            differentialdiffusion_327 = differentialdiffusion.apply(
                model=get_value_at_index(loraloadermodelonly_447, 0)
            )

            modelsamplingflux_320 = modelsamplingflux.patch(
                max_shift=1.2000000000000002,
                base_shift=0.5,
                width=get_value_at_index(layerutility_getimagesize_321, 0),
                height=get_value_at_index(layerutility_getimagesize_321, 1),
                model=get_value_at_index(differentialdiffusion_327, 0),
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

            easy_cleangpuused_416 = easy_cleangpuused.empty_cache(
                anything=get_value_at_index(vaedecode_106, 0),
                unique_id=4304790991534424795,
            )

            growmask_440 = growmask.expand_mask(
                expand=15,
                tapered_corners=True,
                mask=get_value_at_index(layerutility_imagescalebyaspectratio_v2_267, 1),
            )

            maskblur_441 = maskblur.execute(
                amount=15, device="auto", mask=get_value_at_index(growmask_440, 0)
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

            imageupscalewithmodel_435 = imageupscalewithmodel.upscale(
                upscale_model=get_value_at_index(upscalemodelloader_434, 0),
                image=get_value_at_index(imagecompositemasked_439, 0),
            )

            get_image_size_436 = get_image_size.get_size(
                image=get_value_at_index(loadimage_229, 0)
            )

            imagescale_437 = imagescale.upscale(
                upscale_method="bicubic",
                width=get_value_at_index(get_image_size_436, 0),
                height=get_value_at_index(get_image_size_436, 1),
                crop="disabled",
                image=get_value_at_index(imageupscalewithmodel_435, 0),
            )

            growmask_444 = growmask.expand_mask(
                expand=15,
                tapered_corners=True,
                mask=get_value_at_index(loadimagemask_432, 0),
            )

            maskblur_445 = maskblur.execute(
                amount=0, device="auto", mask=get_value_at_index(growmask_444, 0)
            )

            imagecompositemasked_442 = imagecompositemasked.composite(
                x=0,
                y=0,
                resize_source=False,
                destination=get_value_at_index(loadimage_229, 0),
                source=get_value_at_index(imagescale_437, 0),
                mask=get_value_at_index(maskblur_445, 0),
            )

            saveimage_448 = saveimage.save_images(
                filename_prefix="ComfyUI_cloeth",
                images=get_value_at_index(imagecompositemasked_442, 0),
            )


if __name__ == "__main__":
    main()
