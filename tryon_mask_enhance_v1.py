import os
import time
import random
import sys
import copy
from typing import Sequence, Mapping, Any, Union
import torch
import tqdm, time, os
from PIL import Image, ImageChops, ImageDraw
import glob
from itertools import product
from src.util import (
    tensor2pil, pil2tensor, pilmask2tensor, make_image_grid, draw_text, mask2image,
    expand_bbox,get_value_at_index,MaskAdd, paint_bbox_tensor, paint_bbox_pil, pil_resize_with_aspect_ratio
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
from src.sagspeed import zdxApplySageAtt
from src.mask_seg_det import  HumanSegmentParts, FashionSegDetect ,HumanFashionMaskDetailer
from src.yolo_node import UltralyticsInference, UltralyticsVisualization, YOLO, get_yolo_result, yolo_detect
from src.human_mask_detail import HumanMaskSegDetTool
from src.refiner_v1  import Refinerv1
from replace_clothes_with_reference import ReplaceClothesWithReference
import cv2
human_mask_detect_and_expand = HumanMaskSegDetTool.human_mask_detect_and_expand
human_mask_detect_and_expand_with_setting = HumanMaskSegDetTool.human_mask_detect_and_expand_with_setting

# class Refinerv1():
#     pass

DEBUG = True
lego_version = 'tryon_fashion_mask_enhance_v6.0'
base_dir = '/home/dell/study/test_comfy/img/'
record_log = f'{base_dir}/1_{lego_version}_1_a600.txt'
save_dir = f'{base_dir}{lego_version}/'

# /home/dell/study/test_comfy/img/tryon_case_0401
human_cloth_csv = '/home/dell/study/test_comfy/img/ai换装线上case-0401.csv'
import pandas as pd
df = pd.read_csv(human_cloth_csv)
human_position_dc = {}
for _, row in df.iterrows():
    if pd.isna(row['id']):
        continue
    key = int(row['id'])
    position = row['position']
    human_position_dc[key] = position
    human_position_dc[key] = position

cloth_ = base_dir + 'tryon_case_0401/*_cloth_file*'
human_ = base_dir + 'tryon_case_0401/*_file_url*'
mask_ = base_dir + 'tryon_case_0401/*_mask_file*'

clothes = glob.glob(cloth_)
humans = glob.glob(human_)
masks = glob.glob(mask_)
human_dc = {int(i.split('/')[-1].split('_file_url')[0]): i for i in humans }
mask_dc = {int(i.split('/')[-1].split('_mask_file')[0]): i for i in masks }
cloth_dc = {int(i.split('/')[-1].split('_cloth_file')[0]): i for i in clothes }


# 1、 no recongnize, maybe seg humanand rembg then willbe better
MASK_nocover = [9085117, 9083076, 9083313, 9083447, 9085093, 9084804, 9084886, 9083303, 9084642, 9084804,
                9084886, 9085311, 9085940, 9086053, 9086262, 9086449, 9086607]  
MASK_resize_error = [9083772, 9085574, 9085581, 9086482]
MASK_other_area = [9083447, 9084642]
MASK_expandbad = [9084689, 9084689]
MASK_should_human_segment = [9085905, 9086097, 9084197, 9084721, 9086512, 9084721]  # add shoulder  2:exclude hand ,foot
BAD_CASE = MASK_nocover + MASK_resize_error + MASK_other_area  + MASK_expandbad + MASK_should_human_segment
# alread remove  /home/dell/study/test_comfy/img/hard_case_0401

if 0:
    human_clothe_pairs = [
        {'human_id': human_id,  'human_path': human_path, 'cloth_path': cloth_path}
        for human_id, human_path in enumerate(humans)
        for cloth_path in clothes
    ]
human_clothe_pairs = [
    {'human_id': human_id,'human_path': human_path, "human_mask_path":mask_dc[human_id],
     'cloth_path': cloth_dc[human_id],  'position': human_position_dc[human_id]}
    for human_id, human_path in human_dc.items()
    if human_id in human_position_dc  and human_id in BAD_CASE
]
# import shutil
# for it in human_clothe_pairs:
#     src = it['human_path'].split('_file_url')[0] + '*'
#     for it_sub in  glob.glob(src):
#         shutil.copy(it_sub, '/home/dell/study/test_comfy/img/hard_case_0401/')
# mask_pil_im_ = '/home/dell/study/test_comfy/img/human_mask/mask_06.png'

print('>>>>>>>>_humans_cunt:', len(human_clothe_pairs))

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

font_path = '/home/dell/study/test_comfy/wqy-microhei.ttc'
def main():
    import_custom_nodes()

    imageresizekj = NODE_CLASS_MAPPINGS["ImageResizeKJ"]()
    resizemask = NODE_CLASS_MAPPINGS["ResizeMask"]()
    growmask = NODE_CLASS_MAPPINGS["GrowMask"]()
    growmaskwithblur = NODE_CLASS_MAPPINGS["GrowMaskWithBlur"]()
    import folder_paths
    fashion_cls_model = os.path.join(folder_paths.models_dir, "yolo/deepfashion2_yolov8s-seg.pt")
    human_fashion_mask_model = HumanFashionMaskDetailer()
    fashion_detect_model =  FashionSegDetect(fashion_cls_model)
    tryon_processor = ReplaceClothesWithReference()   # faild load some nodes
    
    # convert_masks_to_images = NODE_CLASS_MAPPINGS["Convert Masks to Images"]()

    # start Batch----------
    print('>>>>>>>>>>>>>>> statr geternate')
    start_time = time.time()

    # with tqdm.tqdm(len(human_clothe_pairs)) as bar:
    for inx, human_cloth in enumerate(human_clothe_pairs):

        start_time = time.time()
        # loadimage_229_human_img = loadimage.load_image(image="2.jpg")        
        # loadimagemask_432 = loadimagemask.load_image(image="2_mask.jpg", channel="red")
        human_id = human_cloth['human_id']
        save_img_res = f'{save_dir}{human_id}_{inx}_res_{lego_version}.png'
        print('>> trying to generate : ', save_img_res)

        human_img = Image.open(human_cloth['human_path']).convert('RGB')
        loadimage_229_human_img = (pil2tensor(human_img), )  # B, H, W, C = image.shape  not enough values to unpack (expected 4, got 3)
        # ------------------- wraper as enhance process -----------------
        _, H, W, _ = loadimage_229_human_img[0].shape
        ORIG_BBOX = [0, 0, W, H]
        ORIG_BBOX_NORMAL = [0, 0, 1.0, 1.0]
        human_img_crop_enhanced = human_img
        
        position = None if 'position' not in human_cloth else human_cloth['position']

        loadimagemask_init_432 = None
        if 'human_mask_path' not in human_cloth:
            mask_img = None
            # get human segment mask , better get the down_parts mask
            # human_mask_optimazed = human_mask_detect_and_expand(loadimage_229_human_img[0], human_fashion_mask_model) # [1,2048,1536]
        else:
            # test load a musk see what`s  shape 
            mask_img = Image.open(human_cloth['human_mask_path'])
            loadimagemask_init_432 = (pilmask2tensor(mask_img),)       #  shape be like  1,2560, 1920

        cloth_img = Image.open(human_cloth['cloth_path']).convert('RGB')
        loadimage_228_cloth = (pil2tensor(cloth_img), )

        imageresizekj_398 = imageresizekj.resize( # huaman 等效1K缩放~ 
            width=1536,
            height=1536,
            upscale_method="nearest-exact",
            keep_proportion=True,
            divisible_by=2,
            crop="disabled",
            image=get_value_at_index(loadimage_229_human_img, 0),  # human_image
        )
        human_imageresizekj_by32 = imageresizekj.resize( # huaman 等效1K缩放~ 
            width=1024, height=1024, upscale_method="nearest-exact",
            keep_proportion=True, divisible_by=32, crop="disabled",
            image=get_value_at_index(loadimage_229_human_img, 0),  # human_image
        )

        # human - extract main person
        # human_img = Image.open('/home/dell/study/test_comfy/img/tryon_case_0401/9085117_file_url.jpeg').convert('RGB')
        # loadimage_229_human_img = (pil2tensor(human_img), )
        # human_imageresizekj_by32 = imageresizekj.resize( # huaman 等效1K缩放~ 
        #     width=1536, height=1536, upscale_method="nearest-exact",
        #     keep_proportion=True, divisible_by=32, crop="disabled",
        #     image=get_value_at_index(loadimage_229_human_img, 0),  # human_image
        # )
        # human_img = tensor2pil(human_imageresizekj_by32[0])
        # body_res = yolo_detect(human_imageresizekj_by32[0], detec_type = 'body', debug=True)
        # box_mask = body_res['person'][0]
        # bx = box_mask['bbox_xy']
        # draw = ImageDraw.Draw(human_img)
        # draw.rectangle(bx, fill=(110, 0, 0), width=10)

        # # detetc hand~
        # hand_res = yolo_detect(human_imageresizekj_by32[0], detec_type = 'hand', debug=True)
        # if 'hand' in hand_res:
        #     hand_res_dc = hand_res['hand'][0]
        #     hand_bbox_xy = hand_res_dc['bbox_xy']
        #     draw.rectangle(hand_bbox_xy, fill=(0, 255, 0), width=5)
        # # detetc foot~
        # foot_res = yolo_detect(human_imageresizekj_by32[0], detec_type = 'foot', debug=True)
        # if 'foot' in foot_res:
        #     foot_res_dc = foot_res['foot'][0]
        #     foot_bbox_xy = foot_res_dc['bbox_xy']
        #     draw.rectangle(foot_bbox_xy, fill=(0, 0, 128), width=5)

        # make_image_grid([
        #     human_img, 
        #     body_res['person'][0]['mask'],foot_res['debug_image'],  body_res['debug_image']
        # ], rows=1, cols=4).save('/home/dell/study/test_comfy/tryon_1_person_detect.png')
        # debug to see all result

        # -- human_rembg for better masking ---- 
        with torch.inference_mode():
            human_image_rmbg_398 = tryon_processor.layermask_birefnetultrav2.birefnet_ultra_v2(
                detail_method="VITMatte",
                detail_erode=4,
                detail_dilate=2,
                black_point=0.01,
                white_point=0.99,
                max_megapixels=2,
                process_detail=False,
                device="cuda",
                birefnet_model=get_value_at_index(tryon_processor.layermask_loadbirefnetmodelv2_272, 0),
                image=get_value_at_index(imageresizekj_398, 0),
            )
        human_image_rmbg_398 = (
            tryon_processor.layerutility_imageremovealpha.image_remove_alpha(
                fill_background=True,
                background_color="#FFFFFF",
                RGBA_image=get_value_at_index(human_image_rmbg_398, 0),
            )
        )
        # body_res['debug_image'].save('/home/dell/study/test_comfy/tryon_1_person_detect_no.png')
        body_res = yolo_detect(human_imageresizekj_by32[0], detec_type = 'body', debug=True)
        if body_res is not None and 'person' in body_res:
            box_mask = body_res['person'][0]
            box_mask_mask = box_mask['mask']
            bbox_normal = box_mask['bbox_n']
            # bx = box_mask['bbox_xy']
            # bx_width, bx_height = bx[2] - bx[0], bx[3] - bx[1]; wh_rate = bx_height / bx_width
            # if wh_rate > 1.4:      print('is expand ok?')
            bbox_normal_expand, bbox_expand  = expand_bbox(bbox=bbox_normal, image_width=W, image_height=H, expand_ratio=0.6, width_more=True)
            ORIG_BBOX_NORMAL = bbox_normal_expand
            ORIG_BBOX = [int(x_) for x_ in bbox_expand]
            human_img_crop_enhanced = human_img.crop(ORIG_BBOX)
            with torch.inference_mode():
                human_img_crop_enhanced_rmbg = tryon_processor.layermask_birefnetultrav2.birefnet_ultra_v2(
                    detail_method="VITMatte", detail_erode=4, detail_dilate=2,
                    black_point=0.01, white_point=0.99, max_megapixels=2, process_detail=False, device="cuda",
                    birefnet_model=get_value_at_index(tryon_processor.layermask_loadbirefnetmodelv2_272, 0),
                    image=pil2tensor(human_img_crop_enhanced),
                )
            human_img_crop_enhanced_rmbg = (
                tryon_processor.layerutility_imageremovealpha.image_remove_alpha(
                    fill_background=True,
                    background_color="#FFFFFF",
                    RGBA_image=get_value_at_index(human_img_crop_enhanced_rmbg, 0),
                )
            )
            loadimage_229_human_img_crop_enhanced = (pil2tensor(human_img_crop_enhanced), )  # 输入的图像移除了背景，接下来没法计算了
            imageresizekj_398 = imageresizekj.resize( # huaman 等效1K缩放~ 再处理~
                width=1536,
                height=1356,
                upscale_method="nearest-exact",
                keep_proportion=True,
                divisible_by=32,
                crop="disabled",
                image=get_value_at_index(human_img_crop_enhanced_rmbg, 0),  # human_image
            )
            # -- human_rembg for better masking ---- 
            with torch.inference_mode():
                human_image_rmbg_398 = tryon_processor.layermask_birefnetultrav2.birefnet_ultra_v2(
                    detail_method="VITMatte",
                    detail_erode=4,
                    detail_dilate=2,
                    black_point=0.01,
                    white_point=0.99,
                    max_megapixels=2,
                    process_detail=False,
                    device="cuda",
                    birefnet_model=get_value_at_index(tryon_processor.layermask_loadbirefnetmodelv2_272, 0),
                    image=get_value_at_index(imageresizekj_398, 0),
                )
            human_image_rmbg_398 = (
                tryon_processor.layerutility_imageremovealpha.image_remove_alpha(
                    fill_background=True,
                    background_color="#FFFFFF",
                    RGBA_image=get_value_at_index(human_image_rmbg_398, 0),
                )
            )
            
            new_rate = (bbox_expand[3] - bbox_expand[1]) / (bbox_expand[2] - bbox_expand[0])
            if 0:
                debug_img = body_res['debug_image']
                bbox_ = box_mask['bbox_xy']
                draw = ImageDraw.Draw(debug_img)
                w_,h_ = debug_img.size
                bbox_debug = [bbox_normal_expand[0] * w_, bbox_normal_expand[1] * h_, bbox_normal_expand[2] * w_, bbox_normal_expand[3] * h_]
                draw.rectangle(xy=bbox_debug, width=5)
                draw.rectangle(xy=bbox_, width=10)
                debug_image_collection = [
                    ( human_img.crop(
                            expand_bbox(bbox=bbox_normal, image_width=W, image_height=H, expand_ratio=0.5)[1]
                       ), "test_no_expand_w_crop"
                    ),
                    (human_img.crop(ORIG_BBOX), "test_expand_crop"), 
                    (debug_img, 'res')
                ]
                debug_image_collection = [
                    draw_text(pil_resize_with_aspect_ratio(im_, 2048), txt_) 
                    for im_, txt_ in debug_image_collection
                ]
                make_image_grid(
                    debug_image_collection, cols=3, rows=1
                ).save(f'{save_dir}{human_id}_person_detect_mask_res.png')

        # fashion style  match human mask :，什么衣服，匹配什么amsk~  + 下肢grow
        clothe_resize1k = imageresizekj.resize(
            width=1024, height=1024, upscale_method="nearest-exact",
            keep_proportion=True, divisible_by=32, crop="disabled",
            image=get_value_at_index(loadimage_228_cloth, 0),  # clothe
        )
        fashion_det_res = fashion_detect_model(tensor2pil(clothe_resize1k[0]))  # get fashion type
        
        extra_setting = {} if position == 'whole' else fashion_det_res[1]
        human_mask_result2 = human_fashion_mask_model(human_image_rmbg_398[0], extra_setting=extra_setting)
        loadimagemask_432 = (human_mask_result2[1], )
        mask_img_pil_enhanced = tensor2pil(loadimagemask_432[0])   # for debug
        
        human_fashion_type = fashion_detect_model(tensor2pil(human_image_rmbg_398[0]))  # get human`s fashion type
        description = [
            'human:', ' '.join(human_fashion_type[2]),  'fashion: ', ' '.join(fashion_det_res[2]), 'position:', position
        ]
        description = ' '.join(description)
        print('>> tryon_type_', description)

        #  Max low_cover_mask: test in v2 seemed bad
        box_mask = None
        for fashion_type_ in fashion_det_res[1]:
            if 'down_' in fashion_type_ and position in ['pants', 'whole']:   #  down_short  down_long  down_longlong
                human_mask_result2_big_low_part = human_fashion_mask_model(human_image_rmbg_398[0], extra_setting={'low_cover_big'})
                mask_img_pil = tensor2pil(human_mask_result2_big_low_part[1])  #  mode=L size=1242x204
                box_mask = mask_img_pil.getbbox()
                break
        if box_mask:  # position == 'pants' and cloth is pantas or dress
            box_width = box_mask[2] - box_mask[0]
            box_height = box_mask[3] - box_mask[1]            
            human_mask_growed3 = human_mask_result2_big_low_part[1]
            human_in_pants = 'down_long' in human_fashion_type[1] or 'down_short' in human_fashion_type[1]   # 人穿裤子~
            if human_in_pants and 'down_longlong' in fashion_det_res[1]:     # 要换裙子
                expantd_pixel = int(box_width / 20)  # 有个穿长裤的的小姐姐要穿裙子，就绘bad~  need more big mask
                human_mask_growed3 = growmask.expand_mask(
                    expand=expantd_pixel, tapered_corners=False, mask=human_mask_result2_big_low_part[1],
                )
            else:   # 人本来穿的就是裙子~ ，但是legs 可能没有覆盖到~ 需要做half? 下半身的扩展
                human_mask_growed3 = growmask.expand_mask(
                    expand=5, tapered_corners=False, mask=human_mask_result2_big_low_part[1],
                )
            human_mask_add = MaskAdd().add_masks(human_mask_growed3[0], loadimagemask_432[0])[0]
            loadimagemask_432 = (human_mask_add, )    # (torch.Size([1, 2560, 1920]), )
            mask_img_pil_enhanced = tensor2pil(human_mask_add)

        # - human`s face area excluded
        human_image_enhace_resize_by32 = imageresizekj.resize( # huaman 等效1K缩放~ 
            width=1024, height=1024, upscale_method="nearest-exact", keep_proportion=True,
            divisible_by=32, crop="disabled",
            image=loadimage_229_human_img_crop_enhanced[0]
        )
        face_res = yolo_detect(human_image_enhace_resize_by32[0], detec_type = 'face', debug=True)
        face_new_mask_im = None # for debug
        if 'face' in face_res:
            face_res_dc = face_res['face'][0]
            face_bbox_n = face_res_dc['bbox_n']
            face_bbox = face_res_dc['bbox_xy']
            _, h_, w_= loadimagemask_432[0].shape
            face_bbox_normal_expand, face_bbox_expand  = expand_bbox(bbox=face_bbox_n, image_width=w_, image_height=h_, expand_ratio=0.1)
            new_human_mask_tensor = copy.deepcopy(loadimagemask_432[0])
            new_human_mask_tensor = paint_bbox_tensor(new_human_mask_tensor, face_bbox_expand)
            loadimagemask_432 = (new_human_mask_tensor, )

            if DEBUG:
                new_tensor_im = tensor2pil(new_human_mask_tensor)
                face_new_mask_im = new_tensor_im
        # hand and foot exclude
        if 1:
            hand_res = yolo_detect(human_image_enhace_resize_by32[0], detec_type = 'hand', debug=True)
            if 'hand' in hand_res:
                hand_res_dc = hand_res['hand'][0]
                hand_bbox_n = hand_res_dc['bbox_n']
                _, h_, w_= loadimagemask_432[0].shape
                hand_bbox_normal_expand, hand_bbox_expand  = expand_bbox(bbox=hand_bbox_n, image_width=w_, image_height=h_, expand_ratio=0.1)
                new_human_mask_tensor = copy.deepcopy(loadimagemask_432[0])
                new_human_mask_tensor = paint_bbox_tensor(new_human_mask_tensor, hand_bbox_expand)
                loadimagemask_432 = (new_human_mask_tensor, )

            # detetc foot~
            foot_res = yolo_detect(human_imageresizekj_by32[0], detec_type = 'foot', debug=True)
            if 'foot' in foot_res:
                foot_res_dc = foot_res['foot'][0]
                # foot_bbox_xy = foot_res_dc['bbox_xy']
                foot_bbox_n = foot_res_dc['bbox_n']
                _, h_, w_= loadimagemask_432[0].shape
                foot_bbox_normal_expand, foot_bbox_expand  = expand_bbox(bbox=foot_bbox_n, image_width=w_, image_height=h_, expand_ratio=0.1)
                new_human_mask_tensor = copy.deepcopy(loadimagemask_432[0])
                new_human_mask_tensor = paint_bbox_tensor(new_human_mask_tensor, foot_bbox_expand)
                loadimagemask_432 = (new_human_mask_tensor, )


        # users uploaded and self defined mask
        loadimagemask_init_432_img = tensor2pil(loadimagemask_init_432[0])
        mask_before_add = tensor2pil(loadimagemask_432[0])
        if loadimagemask_init_432 is not None:
            # bug, some time it shrink resized, cant directly add!!!
            _, h_, w_= loadimagemask_init_432[0].shape
            orig_bbox_normal, orig_box  = expand_bbox(bbox=ORIG_BBOX_NORMAL, image_width=w_, image_height=h_, expand_ratio=0)
            orig_box = [int(x_) for x_ in orig_box]
            loadimagemask_init_432_img = tensor2pil(loadimagemask_init_432[0])
            loadimagemask_init_432_img = loadimagemask_init_432_img.crop(orig_box)
            loadimagemask_init_432 = (pil2tensor(loadimagemask_init_432_img), )

            _, h_, w_= loadimagemask_432[0].shape
            loadimagemask_init_432 = resizemask.resize(
                height=h_, width=w_, keep_proportions=False,
                upscale_method="nearest-exact", crop="disabled",
                mask=get_value_at_index(loadimagemask_init_432, 0),
            )
            human_mask_add = MaskAdd().add_masks(loadimagemask_init_432[0], loadimagemask_432[0])[0]
            loadimagemask_432 = (human_mask_add, )    # (torch.Size([1, 2560, 1920]), )
            mask_img_pil_enhanced = tensor2pil(human_mask_add)


        # fashion process: remove head
        if 0:  # remove_fashion head part~ optional
            fashion_face_res = yolo_detect(clothe_resize1k[0], detec_type = 'face', debug=True)
            if 'face' in fashion_face_res:
                fashion_face_res_dc = fashion_face_res['face'][0]
                face_bbox_n = fashion_face_res_dc['bbox_n']
                face_bbox = fashion_face_res_dc['bbox_xy']
                _, h_, w_, _= loadimage_228_cloth[0].shape
                face_bbox_normal_expand, face_bbox_expand  = expand_bbox(bbox=face_bbox_n, image_width=w_, image_height=h_, expand_ratio=0.1)
                new_cloth_tensor = copy.deepcopy(loadimage_228_cloth[0])
                new_cloth_tensor = paint_bbox_tensor(new_cloth_tensor, face_bbox_expand, color=1.0)  # cloth outer area is white
                loadimage_228_cloth = (new_cloth_tensor, )

        # --------------- wrapter end ------------
        # call tryon processor
        if 1:
            tryon_setting = {'tops':False, 'bottoms':False, 'whole':False}
            res_image, result_dc = tryon_processor.forward(
                human_image = loadimage_229_human_img_crop_enhanced, cloth_image = loadimage_228_cloth, 
                human_mask = loadimagemask_432, req_set = tryon_setting
            )
        else:  # faster debug~
            result_dc = {}
            res_image = tensor2pil(loadimage_229_human_img_crop_enhanced[0])
            result_dc['cloth_rmbg'] = cloth_img
            result_dc['middle_lq'] = res_image
            result_dc['refiner_res'] = res_image
            

        # paste to the orig image ---------------- ~~~~~~~ 有时贴回去 不正确--
        final_res_image = copy.deepcopy(human_img)
        final_res_image.paste(res_image, box=ORIG_BBOX)  # TODO , add mask to do it  some time generated res is not right

        # for showing ~
        human_imageresize15k = imageresizekj.resize( # huaman 等效1K缩放~ 再处理~
            width=1536, height=1536, upscale_method="nearest-exact",
            keep_proportion=True, divisible_by=32, crop="disabled",
            image=get_value_at_index(loadimage_229_human_img, 0),  # human_image
        )
        res_image_15k = tensor2pil(
            imageresizekj.resize(
                width=1536, height=1536, upscale_method="nearest-exact",
                keep_proportion=True, divisible_by=32, crop="disabled",
                image=pil2tensor(res_image.convert('RGB')),
            )[0]
        )
        human_img = tensor2pil(human_imageresize15k[0])
        human_mask = tensor2pil(loadimagemask_432[0])
        mask_img_pil1 = tensor2pil(human_mask_result2[1]) # mask notoptimazed
        # cloth_rmbg = tensor2pil(layerutility_imageremovealpha_273_cloth_rmbg[0])
        cloth_rmbg = result_dc['cloth_rmbg']
        repaint_area_res = result_dc['middle_lq']
        refiner_img = result_dc['refiner_res']

        endtime = time.time()
        cost_time = f'{endtime - start_time:.2f}'
        debug_image_collection = [
            (human_img,'human'), (cloth_rmbg, 'cloth'),
            (res_image_15k, f"cost {cost_time} generated"),
            (mask_img_pil1, "human_mask no optmiz"),
            (mask_img_pil_enhanced, description), 
            (mask_before_add, 'mask_before_merge'),
            (loadimagemask_init_432_img, 'mask_uploaded'),
            (human_mask, "human_mask"),
            (final_res_image, "final_img paste back"),
            # draw_text(refiner_img, "refiner"),
        ]
        debug_image_collection = [
            draw_text(pil_resize_with_aspect_ratio(im_, 2048), txt_, font_path=font_path) 
            for im_, txt_ in debug_image_collection
        ]

        debug_img = make_image_grid(debug_image_collection, cols=len(debug_image_collection), rows=1).convert('RGB')
        debug_img.save(f'{save_dir}{human_id}_{inx}_res_{lego_version}.jpeg')
        # debug_imgs = debug_image_collection[:3]+[debug_image_collection[-1]]
        # make_image_grid(
        #     debug_imgs, cols=len(debug_imgs), rows=1
        # ).convert('RGB').save(f'{save_dir}{human_id}_{inx}_res_light_{lego_version}.png')
        # final_res_image.convert('RGB').save(f'{save_dir}final_res_{human_id}_{lego_version}.png')
        # ref.write(f"id:{human_id} _{inx}_ {lego_version} cost: {cost_time} sec\n")
        # bar.update(1)
        # break
        
def test_original():
    import_custom_nodes()

    imageresizekj = NODE_CLASS_MAPPINGS["ImageResizeKJ"]()
    resizemask = NODE_CLASS_MAPPINGS["ResizeMask"]()
    growmask = NODE_CLASS_MAPPINGS["GrowMask"]()
    growmaskwithblur = NODE_CLASS_MAPPINGS["GrowMaskWithBlur"]()
    import folder_paths
    fashion_cls_model = os.path.join(folder_paths.models_dir, "yolo/deepfashion2_yolov8s-seg.pt")
    human_fashion_mask_model = HumanFashionMaskDetailer()
    fashion_detect_model =  FashionSegDetect(fashion_cls_model)
    tryon_processor = ReplaceClothesWithReference()   # faild load some nodes
    
    # convert_masks_to_images = NODE_CLASS_MAPPINGS["Convert Masks to Images"]()

    # start Batch----------
    print('>>>>>>>>>>>>>>> statr geternate')
    start_time = time.time()

    # with tqdm.tqdm(len(human_clothe_pairs)) as bar:
    for inx, human_cloth in enumerate(human_clothe_pairs):

        start_time = time.time()
        # loadimage_229_human_img = loadimage.load_image(image="2.jpg")        
        # loadimagemask_432 = loadimagemask.load_image(image="2_mask.jpg", channel="red")
        human_id = human_cloth['human_id']
        save_img_res = f'{save_dir}{human_id}_{inx}_res_{lego_version}.png'
        print('>> trying to generate : ', save_img_res)

        human_img = Image.open(human_cloth['human_path']).convert('RGB')
        loadimage_229_human_img = (pil2tensor(human_img), )  # B, H, W, C = image.shape  not enough values to unpack (expected 4, got 3)
        # ------------------- wraper as enhance process -----------------
        _, H, W, _ = loadimage_229_human_img[0].shape
        ORIG_BBOX = [0, 0, W, H]
        ORIG_BBOX_NORMAL = [0, 0, 1.0, 1.0]
        human_img_crop_enhanced = human_img
        
        position = None if 'position' not in human_cloth else human_cloth['position']

        loadimagemask_init_432 = None
        if 'human_mask_path' not in human_cloth:
            mask_img = None
            # get human segment mask , better get the down_parts mask
            # human_mask_optimazed = human_mask_detect_and_expand(loadimage_229_human_img[0], human_fashion_mask_model) # [1,2048,1536]
        else:
            # test load a musk see what`s  shape 
            mask_img = Image.open(human_cloth['human_mask_path'])
            loadimagemask_init_432 = (pilmask2tensor(mask_img),)       #  shape be like  1,2560, 1920

        cloth_img = Image.open(human_cloth['cloth_path']).convert('RGB')
        loadimage_228_cloth = (pil2tensor(cloth_img), )

        # call tryon processor
        tryon_setting = {'tops':False, 'bottoms':False, 'whole':False}
        res_image, result_dc = tryon_processor.forward(
            human_image = loadimage_229_human_img, cloth_image = loadimage_228_cloth, 
            human_mask = loadimagemask_init_432, req_set = tryon_setting
        )

        # paste to the orig image ---------------- ~~~~~~~
        final_res_image = res_image

        human_img = tensor2pil(loadimage_229_human_img[0])
        human_mask = tensor2pil(loadimagemask_init_432[0])
        # cloth_rmbg = tensor2pil(layerutility_imageremovealpha_273_cloth_rmbg[0])
        cloth_rmbg = result_dc['cloth_rmbg']
        repaint_area_res = result_dc['middle_lq']
        refiner_img = result_dc['refiner_res']

        last_version = 'tryon_fashion_mask_enhance_v5.5'
        last_version_res = f'/home/dell/study/test_comfy/img/{last_version}/final_res_{human_id}_{last_version}.png'
        last_version_res = Image.open(last_version_res)

        endtime = time.time()
        cost_time = f'{endtime - start_time:.2f}'
        debug_image_collection = [
            (human_img,'human'), (cloth_rmbg, 'cloth'),
            (final_res_image, f"cost {cost_time} old_tryon"),
            (last_version_res, "new_enhance tryon"),
            (human_mask, "human_mask"),
            # draw_text(refiner_img, "refiner"),
        ]
        debug_image_collection = [
            draw_text(pil_resize_with_aspect_ratio(im_, 2048), txt_, font_path=font_path) 
            for im_, txt_ in debug_image_collection
        ]

        debug_img = make_image_grid(debug_image_collection, cols=len(debug_image_collection), rows=1).convert('RGB')
        debug_img.save(f'{save_dir}{human_id}_{inx}_res_{lego_version}.jpeg')
        # final_res_image.convert('RGB').save(f'{save_dir}final_res_{human_id}_{lego_version}.png')
        # ref.write(f"id:{human_id} _{inx}_ {lego_version} cost: {cost_time} sec\n")
        # bar.update(1)
        # break

if __name__ == '__main__':
    print('>> run main or  refiner?  commnet this line!')
    if 1:
        main()
    elif 0:
        test_original()
    elif 0:
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
    else:
        imageresizekj = NODE_CLASS_MAPPINGS["ImageResizeKJ"]()
        human_path = human_clothe_pairs[1]['human_path']
        human_im = Image.open(human_path).resize((640, 768))
        human_im_1 = (pil2tensor(human_im), )
        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_228_cloth = loadimage.load_image(image="cloth_02.png")
        human_imageresizekj_398 = imageresizekj.resize( # huaman 等效1K缩放~ 再处理~
            width=1024,
            height=1024,
            upscale_method="nearest-exact",
            keep_proportion=True,
            divisible_by=2,
            crop="disabled",
            image=get_value_at_index(human_im_1, 0),  # human_image
        )

        import folder_paths
        yolo_models = glob.glob(os.path.join(folder_paths.models_dir, "yolo/*.pt"))
        person_model = YOLO(os.path.join(folder_paths.models_dir, "yolo/person_yolov8m-seg.pt"))
        yolo_infer = UltralyticsInference()
        yolo_viser = UltralyticsVisualization()
        
        yres = yolo_infer.inference(model=person_model, image=human_im_1[0], classes="None")
        yolo_vis1 = tensor2pil(
            yolo_viser.visualize(human_im_1[0], yres[0])[0]
        )
        yolo_vis1.save('/home/dell/study/test_comfy/img/yolo_person_detect_mask_res.png')  # for debug
        person_res = get_yolo_result(yres[0])   # get label -> [mask, bbox]
        # bbox_expand = expand_face_box(bbox, width, height)
        face_res = yolo_detect(human_im_1[0], detec_type = 'face', debug=False)
        body_res = yolo_detect(human_im_1[0], detec_type = 'body', debug=False)
