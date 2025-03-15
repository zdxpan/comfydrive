import os
import random
import torch
import numpy as np

from nodes import NODE_CLASS_MAPPINGS
from comfyui_workflows.comfyui_tools import get_value_at_index
from src.utils.aliyun_v2 import AliyunOss
from comfyui_workflows.utils import ServiceTemplate
from folder_paths import base_path


class ReplaceClothesWithReference(ServiceTemplate):
    def __init__(self, save_format):
        self.name = self.__class__.__name__
        super(ReplaceClothesWithReference, self).__init__(save_format)
        with torch.inference_mode():
            self.clipvisionloader = NODE_CLASS_MAPPINGS["CLIPVisionLoader"]()
            self.clipvisionloader_329 = self.clipvisionloader.load_clip(
                clip_name="sigclip_vision_patch14_384.safetensors"
            )
            self.layermask_loadbenmodel = NODE_CLASS_MAPPINGS["LayerMask: LoadBenModel"]()
            self.layermask_loadbenmodel_435 = self.layermask_loadbenmodel.load_ben_model(
                model="BEN_Base.pth", device="cuda"
            )
            self.loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
            self.imageresizekj = NODE_CLASS_MAPPINGS["ImageResizeKJ"]()
            self.loadimagemask = NODE_CLASS_MAPPINGS["LoadImageMask"]()
            self.layermask_segformerb2clothesultra = NODE_CLASS_MAPPINGS[
                "LayerMask: SegformerB2ClothesUltra"
            ]()
            self.easy_prompt = NODE_CLASS_MAPPINGS["easy prompt"]()

            self.layermask_segmentanythingultra_v2 = NODE_CLASS_MAPPINGS[
                "LayerMask: SegmentAnythingUltra V2"
            ]()
            self.maskcomposite = NODE_CLASS_MAPPINGS["MaskComposite"]()
            self.switch_mask_crystools = NODE_CLASS_MAPPINGS["Switch mask [Crystools]"]()
            self.growmaskwithblur = NODE_CLASS_MAPPINGS["GrowMaskWithBlur"]()

            self.layerutility_imagescalebyaspectratio_v2 = NODE_CLASS_MAPPINGS[
                "LayerUtility: ImageScaleByAspectRatio V2"
            ]()
            self.layerutility_imagemaskscaleas = NODE_CLASS_MAPPINGS[
                "LayerUtility: ImageMaskScaleAs"
            ]()
            self.layermask_benultra = NODE_CLASS_MAPPINGS["LayerMask: BenUltra"]()

            self.layerutility_imageremovealpha = NODE_CLASS_MAPPINGS[
                "LayerUtility: ImageRemoveAlpha"
            ]()
            self.clipvisionencode = NODE_CLASS_MAPPINGS["CLIPVisionEncode"]()
            self.unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
            self.unetloader_326 = self.unetloader.load_unet(
                unet_name="FLUX.1-Fill-dev_fp8.safetensors", weight_dtype="fp8_e4m3fn"
            )
            self.dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
            self.dualcliploader_322 = self.dualcliploader.load_clip(
                clip_name1="t5xxl_fp16.safetensors",
                clip_name2="clip_l.safetensors",
                type="flux",
            )
            self.loraloader = NODE_CLASS_MAPPINGS["LoraLoader"]()
            self.loraloader_439 = self.loraloader.load_lora(
                lora_name="FLUX.1-Turbo-Alpha.safetensors",
                strength_model=1,
                strength_clip=1,
                model=get_value_at_index(self.unetloader_326, 0),
                clip=get_value_at_index(self.dualcliploader_322, 0),
            )
            self.cliptextencodeflux = NODE_CLASS_MAPPINGS["CLIPTextEncodeFlux"]()
            self.vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
            self.vaeloader = self.vaeloader.load_vae(vae_name="ae.safetensors")
            self.easy_imageconcat = NODE_CLASS_MAPPINGS["easy imageConcat"]()
            self.solidmask = NODE_CLASS_MAPPINGS["SolidMask"]()
            self.masktoimage = NODE_CLASS_MAPPINGS["MaskToImage"]()
            self.imagetomask = NODE_CLASS_MAPPINGS["ImageToMask"]()
            self.inpaintmodelconditioning = NODE_CLASS_MAPPINGS["InpaintModelConditioning"]()
            self.stylemodelloader = NODE_CLASS_MAPPINGS["StyleModelLoader"]()
            self.stylemodelloader_330 = self.stylemodelloader.load_style_model(
                style_model_name="flux1-redux-dev.safetensors"
            )
            self.upscalemodelloader = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
            self.layerutility_getimagesize = NODE_CLASS_MAPPINGS["LayerUtility: GetImageSize"]()
            self.differentialdiffusion = NODE_CLASS_MAPPINGS["DifferentialDiffusion"]()
            self.modelsamplingflux = NODE_CLASS_MAPPINGS["ModelSamplingFlux"]()
            self.stylemodelapply = NODE_CLASS_MAPPINGS["StyleModelApply"]()
            self.fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
            self.ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
            self.vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
            self.easy_imagesplitgrid = NODE_CLASS_MAPPINGS["easy imageSplitGrid"]()
            self.easy_imagessplitimage = NODE_CLASS_MAPPINGS["easy imagesSplitImage"]()
            self.get_image_size = NODE_CLASS_MAPPINGS["Get Image Size"]()
            self.imagescale = NODE_CLASS_MAPPINGS["ImageScale"]()
            self.eg_zz_bsyh = NODE_CLASS_MAPPINGS["EG_ZZ_BSYH"]()
            self.jwmaskresize = NODE_CLASS_MAPPINGS["JWMaskResize"]()
            self.imagecompositemasked = NODE_CLASS_MAPPINGS["ImageCompositeMasked"]()
        print(f"=========={self.name} loaded==========")


    def forward(self, request, time_string):
        init_image_file, \
        mask_image_file, \
        _, \
        userdefined_images, \
        _, \
        _, \
        _, \
        _, \
        _, \
        user_id, \
        _ = self.input_from_request(request)
        # data_json = request.get_json()
        data_json = request if isinstance(request, dict) else request.get_json()
        tops = data_json.get('tops')
        bottoms = data_json.get('bottoms')
        whole = data_json.get('whole')

        results = []
        with torch.inference_mode():
            for q in range(1):
                if tops:
                    seg_prompt = "upper body clothing"
                    if not userdefined_images:
                        userdefined_images = f'{base_path}/comfyui_workflows/workflows/clothes/tops/{tops}.jpg'
                if bottoms:
                    seg_prompt = "lower body clothing"
                    if not userdefined_images:
                        userdefined_images = f'{base_path}/comfyui_workflows/workflows/clothes/bottoms/{bottoms}.jpg'
                if whole:
                    seg_prompt = "whole body clothing"
                    if not userdefined_images:
                        userdefined_images = f'{base_path}/comfyui_workflows/workflows/clothes/whole/{whole}.jpg'

                loadimage_229 = self.loadimage.load_image(image=init_image_file)
                loadimage_228 = self.loadimage.load_image(image=userdefined_images)
                self.imageresizekj_398 = self.imageresizekj.resize(
                    width=1024,
                    height=1024,
                    upscale_method="nearest-exact",
                    keep_proportion=True,
                    divisible_by=2,
                    crop="disabled",
                    image=get_value_at_index(loadimage_229, 0),
                )

                maskcomposite_432 = None
                loadimagemask_471 = None
                if mask_image_file:
                    loadimagemask_471 = self.loadimagemask.load_image(
                        image=mask_image_file, channel="red"
                    )
                    jwmaskresize_474 = self.jwmaskresize.execute(
                        height=get_value_at_index(self.imageresizekj_398, 2),
                        width=get_value_at_index(self.imageresizekj_398, 1),
                        interpolation_mode="bicubic",
                        mask=get_value_at_index(loadimagemask_471, 0),
                    )
                else:
                    easy_prompt_430 = self.easy_prompt.doit(
                        prompt=seg_prompt, main="none", lighting="none"
                    )
                    layermask_segformerb2clothesultra_431 = (
                        self.layermask_segformerb2clothesultra.segformer_ultra(
                            face=False,
                            hair=False,
                            hat=False,
                            sunglass=False,
                            left_arm=False,
                            right_arm=False,
                            left_leg=False,
                            right_leg=False,
                            upper_clothes=True if tops or whole else False,
                            skirt=True if bottoms or whole else False,
                            pants=True if bottoms or whole else False,
                            dress=True if bottoms or whole else False,
                            belt=True if bottoms or whole else False,
                            shoe=False,
                            bag=False,
                            scarf=False,
                            detail_method="VITMatte",
                            detail_erode=12,
                            detail_dilate=6,
                            black_point=0.15,
                            white_point=0.99,
                            process_detail=True,
                            device="cuda",
                            max_megapixels=2,
                            image=get_value_at_index(self.imageresizekj_398, 0),
                        )
                    )
                    layermask_segmentanythingultra_v2_399 = (
                        self.layermask_segmentanythingultra_v2.segment_anything_ultra_v2(
                            sam_model="sam_vit_h (2.56GB)",
                            grounding_dino_model="GroundingDINO_SwinT_OGC (694MB)",
                            threshold=0.45,
                            detail_method="VITMatte",
                            detail_erode=6,
                            detail_dilate=6,
                            black_point=0.15,
                            white_point=0.99,
                            process_detail=True,
                            prompt=get_value_at_index(easy_prompt_430, 0),
                            device="cuda",
                            max_megapixels=2,
                            cache_model=False,
                            image=get_value_at_index(self.imageresizekj_398, 0),
                        )
                    )
                    maskcomposite_432 = self.maskcomposite.combine(
                        x=0,
                        y=0,
                        operation="add",
                        destination=get_value_at_index(layermask_segformerb2clothesultra_431, 1),
                        source=get_value_at_index(layermask_segmentanythingultra_v2_399, 1),
                    )


                # self.switch_mask_crystools_473 = self.switch_mask_crystools.execute(
                #     boolean=False,
                #     on_true=get_value_at_index(self.jwmaskresize_474, 0),
                #     on_false=get_value_at_index(self.maskcomposite_432, 0),
                # )
                switch_mask_crystools_473 =  jwmaskresize_474 if mask_image_file else  maskcomposite_432
                
                growmaskwithblur_337 = self.growmaskwithblur.expand_mask(
                    expand=15,
                    incremental_expandrate=0,
                    tapered_corners=True,
                    flip_input=False,
                    blur_radius=10,
                    lerp_alpha=1,
                    decay_factor=1,
                    fill_holes=False,
                    mask=get_value_at_index(switch_mask_crystools_473, 0),
                )
                layerutility_imagescalebyaspectratio_v2_267 = (
                    self.layerutility_imagescalebyaspectratio_v2.image_scale_by_aspect_ratio(
                        aspect_ratio="original",
                        proportional_width=1,
                        proportional_height=1,
                        fit="crop",
                        method="lanczos",
                        round_to_multiple="8",
                        scale_to_side="longest",
                        scale_to_length=1280,
                        background_color="#000000",
                        image=get_value_at_index(self.imageresizekj_398, 0),
                        mask=get_value_at_index(growmaskwithblur_337, 0),
                    )
                )
                layerutility_imagemaskscaleas_268 = (
                    self.layerutility_imagemaskscaleas.image_mask_scale_as(
                        fit="letterbox",
                        method="lanczos",
                        scale_as=get_value_at_index(
                            layerutility_imagescalebyaspectratio_v2_267, 0
                        ),
                        image=get_value_at_index(loadimage_228, 0),
                    )
                )
                layermask_benultra_433 = self.layermask_benultra.ben_ultra_v2(
                    detail_method="VITMatte",
                    detail_erode=4,
                    detail_dilate=2,
                    black_point=0.01,
                    white_point=0.99,
                    max_megapixels=2,
                    process_detail=False,
                    ben_model=get_value_at_index(self.layermask_loadbenmodel_435, 0),
                    image=get_value_at_index(layerutility_imagemaskscaleas_268, 0),
                )
                self.cleangpu(self.layermask_loadbenmodel_435)
                layerutility_imageremovealpha_273 = (
                    self.layerutility_imageremovealpha.image_remove_alpha(
                        fill_background=True,
                        background_color="#FFFFFF",
                        RGBA_image=get_value_at_index(layermask_benultra_433, 0),
                    )
                )
                clipvisionencode_172 = self.clipvisionencode.encode(
                    clip_vision=get_value_at_index(self.clipvisionloader_329, 0),
                    image=get_value_at_index(layerutility_imageremovealpha_273, 0),
                )
                self.cleangpu(self.clipvisionloader_329)
                cliptextencodeflux_323 = self.cliptextencodeflux.encode(
                    clip_l="", t5xxl="", guidance=30, clip=get_value_at_index(self.loraloader_439, 1)
                )
                cliptextencodeflux_325 = self.cliptextencodeflux.encode(
                    clip_l="", t5xxl="", guidance=30, clip=get_value_at_index(self.loraloader_439, 1)
                )
                easy_imageconcat_275 = self.easy_imageconcat.concat(
                    direction="right",
                    match_image_size=False,
                    image1=get_value_at_index(layerutility_imageremovealpha_273, 0),
                    image2=get_value_at_index(layerutility_imagescalebyaspectratio_v2_267, 0),
                )
                solidmask_278 = self.solidmask.solid(
                    value=0,
                    width=get_value_at_index(layerutility_imagescalebyaspectratio_v2_267, 3),
                    height=get_value_at_index(layerutility_imagescalebyaspectratio_v2_267, 4),
                )
                masktoimage_281 = self.masktoimage.mask_to_image(
                    mask=get_value_at_index(solidmask_278, 0)
                )
                masktoimage_282 = self.masktoimage.mask_to_image(
                    mask=get_value_at_index(layerutility_imagescalebyaspectratio_v2_267, 1)
                )
                easy_imageconcat_280 = self.easy_imageconcat.concat(
                    direction="right",
                    match_image_size=False,
                    image1=get_value_at_index(masktoimage_281, 0),
                    image2=get_value_at_index(masktoimage_282, 0),
                )
                imagetomask_283 = self.imagetomask.image_to_mask(
                    channel="red", image=get_value_at_index(easy_imageconcat_280, 0)
                )
                inpaintmodelconditioning_220 = self.inpaintmodelconditioning.encode(
                    positive=get_value_at_index(cliptextencodeflux_323, 0),
                    negative=get_value_at_index(cliptextencodeflux_325, 0),
                    vae=get_value_at_index(self.vaeloader, 0),
                    pixels=get_value_at_index(easy_imageconcat_275, 0),
                    mask=get_value_at_index(imagetomask_283, 0),
                )

                layerutility_getimagesize_321 = self.layerutility_getimagesize.get_image_size(
                    image=get_value_at_index(easy_imageconcat_275, 0)
                )

                differentialdiffusion_327 = self.differentialdiffusion.apply(
                    model=get_value_at_index(self.loraloader_439, 0)
                )

                modelsamplingflux_320 = self.modelsamplingflux.patch(
                    max_shift=1.1500000000000001,
                    base_shift=0.5,
                    width=get_value_at_index(layerutility_getimagesize_321, 0),
                    height=get_value_at_index(layerutility_getimagesize_321, 1),
                    model=get_value_at_index(differentialdiffusion_327, 0),
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

                get_image_size_444 = self.get_image_size.get_size(
                    image=get_value_at_index(loadimage_229, 0)
                )

                imagescale_465 = self.imagescale.upscale(
                    upscale_method="lanczos",
                    width=get_value_at_index(get_image_size_444, 0),
                    height=get_value_at_index(get_image_size_444, 1),
                    crop="disabled",
                    image=get_value_at_index(easy_imagessplitimage_299, 1),
                )

                eg_zz_bsyh_470 = self.eg_zz_bsyh.gaussian_blur_edge(
                    kernel_size=11,
                    sigma=10,
                    shrink_pixels=0,
                    expand_pixels=0,
                    mask=get_value_at_index(layerutility_imagescalebyaspectratio_v2_267, 1),
                )

                jwmaskresize_466 = self.jwmaskresize.execute(
                    height=get_value_at_index(get_image_size_444, 1),
                    width=get_value_at_index(get_image_size_444, 0),
                    interpolation_mode="bicubic",
                    mask=get_value_at_index(eg_zz_bsyh_470, 0),
                )

                imagecompositemasked_448 = self.imagecompositemasked.composite(
                    x=0,
                    y=0,
                    resize_source=False,
                    destination=get_value_at_index(loadimage_229, 0),
                    source=get_value_at_index(imagescale_465, 0),
                    mask=get_value_at_index(jwmaskresize_466, 0),
                )

        results.append((imagecompositemasked_448[0][0].numpy() * 255).astype(np.int32))
        oss_url = [f"x/prod/{self.name}/{user_id}-{time_string}.jpg"]
        return results, oss_url
