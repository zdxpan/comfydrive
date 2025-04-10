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
lego_version = 'tryon_fashion_mask_enhance_v5.4'
base_dir = '/home/dell/study/test_comfy/img/'
record_log = f'{base_dir}/1_{lego_version}_1_a600.txt'
save_dir = f'{base_dir}{lego_version}/'

human_cloth_csv = '/home/dell/study/test_comfy/img/ai换装线上case-0401.csv'
import pandas as pd
df = pd.read_csv(human_cloth_csv)
human_position_dc = {}
for _, row in df.iterrows():
    key = f"{int(row['id'])}"
    position = row['position']
    human_position_dc[key] = position

cloth_ = base_dir + 'tryon_no_mask/cloth_*'
human_ = base_dir + 'tryon_no_mask/image_*'
clothes = glob.glob(cloth_)
humans = glob.glob(human_)
human_dc = {int(i.split('image_')[-1].split('.')[0]): i for i in humans }
cloth_dc = {int(i.split('cloth_')[-1].split('.')[0]): i for i in clothes }

if 0:
    human_clothe_pairs = [
        {'human_id': human_id,  'human_path': human_path, 'cloth_path': cloth_path}
        for human_id, human_path in enumerate(humans)
        for cloth_path in clothes
    ]
human_clothe_pairs = [
    {'human_id': human_id,'human_path': human_path,'cloth_path': cloth_dc[human_id],  'position': human_position_dc[human_id]}
    for human_id, human_path in human_dc.items()
    # if human_id in [2]
]

# mask_pil_im_ = '/home/dell/study/test_comfy/img/human_mask/mask_06.png'

print('>>>>>>>>_humans_cunt:', len(human_clothe_pairs))

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def main():
    import_custom_nodes()

    imageresizekj = NODE_CLASS_MAPPINGS["ImageResizeKJ"]()
    growmask = NODE_CLASS_MAPPINGS["GrowMask"]()
    growmaskwithblur = NODE_CLASS_MAPPINGS["GrowMaskWithBlur"]()
    import folder_paths
    fashion_cls_model = os.path.join(folder_paths.models_dir, "yolo/deepfashion2_yolov8s-seg.pt")
    human_fashion_mask_model = HumanFashionMaskDetailer()
    fashion_detect_model =  FashionSegDetect(fashion_cls_model)
    tryon_processor = ReplaceClothesWithReference()   # faild load some nodes
    
    # start Batch----------
    print('>>>>>>>>>>>>>>> statr geternate')
    start_time = time.time()

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
        
        if 'human_mask_path' not in human_cloth:
            mask_img = None
            loadimagemask_432 = None
            # get human segment mask , better get the down_parts mask
            # human_mask_optimazed = human_mask_detect_and_expand(loadimage_229_human_img[0], human_fashion_mask_model) # [1,2048,1536]
            # loadimagemask_432 = human_mask_optimazed
        else:
            # test load a musk see what`s  shape 
            mask_img = Image.open(human_cloth['human_mask_path'])
            loadimagemask_432 = (pilmask2tensor(mask_img),)       #  shape be like  1,2560, 1920

        cloth_img = Image.open(human_cloth['cloth_path']).convert('RGB')
        loadimage_228_cloth = (pil2tensor(cloth_img), )

        imageresizekj_398 = imageresizekj.resize( # huaman 等效1K缩放~ 
            width=1024,
            height=1024,
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
        # human - extract main persion
        body_res = yolo_detect(human_imageresizekj_by32[0], detec_type = 'body', debug=True)
        if body_res is not None and 'person' in body_res:
            box_mask = body_res['person']
            box_mask_mask = box_mask['mask']
            bbox_normal = box_mask['bbox_n']
            # bx = box_mask['bbox_xy']
            # bx_width, bx_height = bx[2] - bx[0], bx[3] - bx[1]
            # wh_rate = bx_height / bx_width
            # if wh_rate > 1.4:
            #     print('is expand ok?')
            bbox_normal_expand, bbox_expand  = expand_bbox(bbox=bbox_normal, image_width=W, image_height=H, expand_ratio=0.6, width_more=True)
            ORIG_BBOX_NORMAL = bbox_normal_expand
            ORIG_BBOX = [int(x_) for x_ in bbox_expand]
            human_img_crop_enhanced = human_img.crop(ORIG_BBOX)
            loadimage_229_human_img_crop_enhanced = (pil2tensor(human_img_crop_enhanced), )
            imageresizekj_398 = imageresizekj.resize( # huaman 等效1K缩放~ 再处理~
                width=1024,
                height=1024,
                upscale_method="nearest-exact",
                keep_proportion=True,
                divisible_by=2,
                crop="disabled",
                image=get_value_at_index(loadimage_229_human_img_crop_enhanced, 0),  # human_image
            )
            new_rate = (bbox_expand[3] - bbox_expand[1]) / (bbox_expand[2] - bbox_expand[0])
            if DEBUG:
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
        
        human_mask_result2 = human_fashion_mask_model(imageresizekj_398[0], extra_setting=fashion_det_res[1])
        loadimagemask_432 = (human_mask_result2[1], )
        mask_img_pil_enhanced = tensor2pil(loadimagemask_432[0])   # for debug
        
        human_fashion_type = fashion_detect_model(tensor2pil(human_imageresizekj_by32[0]))  # get human`s fashion type
        print('>>  human_fashion_type:', human_fashion_type,  'fashion_type: ', fashion_det_res)

        #  Max low_cover_mask: test in v2 seemed bad
        box_mask = None
        for fashion_type_ in fashion_det_res[1]:
            if 'down_' in fashion_type_:   #  down_short  down_long  down_longlong
                human_mask_result2_big_low_part = human_fashion_mask_model(imageresizekj_398[0], extra_setting={'low_cover_big'})
                mask_img_pil = tensor2pil(human_mask_result2_big_low_part[1])  #  mode=L size=1242x204
                box_mask = mask_img_pil.getbbox()
                break
        if box_mask:
            box_width = box_mask[2] - box_mask[0]
            box_height = box_mask[3] - box_mask[1]            
            human_mask_growed3 = human_mask_result2_big_low_part[1]
            human_in_pants = 'down_long' in human_fashion_type[1] or 'down_short' in human_fashion_type[1]   # 人穿裤子~
            if human_in_pants and 'down_longlong' in fashion_det_res[1]:     # 要换裙子
                expantd_pixel = int(box_width / 15)
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
            face_res_dc = face_res['face']
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

            
        # fashion process: remove head
        if 0:  # remove_fashion head part~ optional
            fashion_face_res = yolo_detect(clothe_resize1k[0], detec_type = 'face', debug=True)
            if 'face' in fashion_face_res:
                fashion_face_res_dc = fashion_face_res['face']
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
            

        # paste to the orig image ---------------- ~~~~~~~
        final_res_image = copy.deepcopy(human_img)
        final_res_image.paste(res_image, box=ORIG_BBOX)

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
        description = list(human_fashion_type[1])[0] + '  ' + list(fashion_det_res[1])[0]
        debug_image_collection = [
            (human_img,'human'), (cloth_rmbg, 'cloth'),
            (res_image_15k, f"cost {cost_time} generated"),
            (mask_img_pil1, "human_mask no optmiz"),
            (mask_img_pil_enhanced, description), 
            (human_mask, "human_mask"),
            (final_res_image, "final_img paste back"),
            # draw_text(refiner_img, "refiner"),
        ]
        debug_image_collection = [
            draw_text(pil_resize_with_aspect_ratio(im_, 2048), txt_) 
            for im_, txt_ in debug_image_collection
        ]

        debug_img = make_image_grid(debug_image_collection, cols=len(debug_image_collection), rows=1).convert('RGB')
        debug_img.save(f'{save_dir}{human_id}_{inx}_res_{lego_version}.png')
        # ref.write(f"id:{human_id} _{inx}_ {lego_version} cost: {cost_time} sec\n")
        # bar.update(1)
        # break
    
def test_original_compare():
    import_custom_nodes()
    imageresizekj = NODE_CLASS_MAPPINGS["ImageResizeKJ"]()
    growmask = NODE_CLASS_MAPPINGS["GrowMask"]()
    growmaskwithblur = NODE_CLASS_MAPPINGS["GrowMaskWithBlur"]()
    import folder_paths
    fashion_cls_model = os.path.join(folder_paths.models_dir, "yolo/deepfashion2_yolov8s-seg.pt")
    tryon_processor = ReplaceClothesWithReference()   # faild load some nodes
    
    # start Batch----------
    print('>>>>>>>>>>>>>>> statr geternate')
    start_time = time.time()

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
        
        if 'human_mask_path' not in human_cloth:
            mask_img = None
            loadimagemask_432 = None
            # get human segment mask , better get the down_parts mask
            # human_mask_optimazed = human_mask_detect_and_expand(loadimage_229_human_img[0], human_fashion_mask_model) # [1,2048,1536]
            # loadimagemask_432 = human_mask_optimazed
        else:
            # test load a musk see what`s  shape 
            mask_img = Image.open(human_cloth['human_mask_path'])
            loadimagemask_432 = (pilmask2tensor(mask_img),)       #  shape be like  1,2560, 1920

        cloth_img = Image.open(human_cloth['cloth_path']).convert('RGB')
        loadimage_228_cloth = (pil2tensor(cloth_img), )

        # call tryon processor
        if 1:
            tryon_setting = {'tops':False, 'bottoms':False, 'whole':False}
            res_image, result_dc = tryon_processor.forward(
                human_image = loadimage_229_human_img, cloth_image = loadimage_228_cloth, 
                human_mask = loadimagemask_432, req_set = tryon_setting
            )
        else:  # faster debug~
            result_dc = {}
            # res_image = tensor2pil(loadimage_229_human_img_crop_enhanced[0])
            result_dc['cloth_rmbg'] = cloth_img
            result_dc['middle_lq'] = res_image
            result_dc['refiner_res'] = res_image
            
        # paste to the orig image ---------------- ~~~~~~~
        # final_res_image = copy.deepcopy(human_img)
        # final_res_image.paste(res_image, box=ORIG_BBOX)
        final_res_image = res_image

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
        description = list(human_fashion_type[1])[0] + '  ' + list(fashion_det_res[1])[0]
        debug_image_collection = [
            (human_img,'human'), (cloth_rmbg, 'cloth'),
            (res_image_15k, f"cost {cost_time} generated"),
            (mask_img_pil1, "human_mask no optmiz"),
            (mask_img_pil_enhanced, description), 
            (human_mask, "human_mask"),
            (final_res_image, "final_img paste back"),
            # draw_text(refiner_img, "refiner"),
        ]
        debug_image_collection = [
            draw_text(pil_resize_with_aspect_ratio(im_, 2048), txt_) 
            for im_, txt_ in debug_image_collection
        ]

        debug_img = make_image_grid(debug_image_collection, cols=len(debug_image_collection), rows=1).convert('RGB')
        debug_img.save(f'{save_dir}{human_id}_{inx}_res_{lego_version}.png')

if __name__ == '__main__':
    print('>> run main or  refiner?  commnet this line!')
    if 1:
        main()
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
        imageresizekj_398 = imageresizekj.resize( # huaman 等效1K缩放~ 再处理~
            width=1024,
            height=1024,
            upscale_method="nearest-exact",
            keep_proportion=True,
            divisible_by=2,
            crop="disabled",
            image=get_value_at_index(human_im_1, 0),  # human_image
        )

        import folder_paths
        yolo_models = glob.glob(os.path.join(folder_paths.models_dir, "yolo/*.pt")
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
