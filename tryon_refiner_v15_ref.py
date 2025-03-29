import os
import time
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import tqdm, time, os
from PIL import Image, ImageChops, ImageDraw
import glob
from itertools import product
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
from src.sagspeed import zdxApplySageAtt
from src.mask_seg_det import  HumanSegmentParts, FashionSegDetect ,HumanFashionMaskDetailer
from src.human_mask_detail import HumanMaskSegDetTool
human_mask_detect_and_expand = HumanMaskSegDetTool.human_mask_detect_and_expand
human_mask_detect_and_expand_with_setting = HumanMaskSegDetTool.human_mask_detect_and_expand_with_setting

from src.refiner_v1  import Refinerv1
from replace_clothes_with_reference import ReplaceClothesWithReference

# class Refinerv1():
#     pass


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

    imageresizekj = NODE_CLASS_MAPPINGS["ImageResizeKJ"]()
    growmask = NODE_CLASS_MAPPINGS["GrowMask"]()
    growmaskwithblur = NODE_CLASS_MAPPINGS["GrowMaskWithBlur"]()

    fashion_cls_model = '/home/dell/study/comfyui/models/yolo/deepfashion2_yolov8s-seg.pt'
    human_fashion_mask_model = HumanFashionMaskDetailer()
    fashion_detect_model =  FashionSegDetect(fashion_cls_model)
    tryon_processor = ReplaceClothesWithReference()
    
    # start Batch----------
    print('>>>>>>>>>>>>>>> statr geternate')
    start_time = time.time()

    # with tqdm.tqdm(len(human_clothe_pairs)) as bar:
    for inx, human_cloeth in enumerate(human_clothe_pairs):

        start_time = time.time()
        # loadimage_229_human_img = loadimage.load_image(image="2.jpg")        
        # loadimagemask_432 = loadimagemask.load_image(image="2_mask.jpg", channel="red")
        # loadimage_228_cloth = loadimage.load_image(image="cloth_02.png")
        human_id = human_cloeth['human_id']
        save_img_res = f'{save_dir}{human_id}_{inx}_res_{lego_version}.png'
        print('>> trying to generate : ', save_img_res)
        # if os.path.exists(save_img_res):
        #     bar.update(1)
        #     continue

        human_img = Image.open(human_cloeth['human_path'])
        loadimage_229_human_img = (pil2tensor(human_img), )  # B, H, W, C = image.shape  not enough values to unpack (expected 4, got 3)
        
        if 'human_mask_path' not in human_cloeth:
            mask_img = None
            loadimagemask_432 = None
            # get human segment mask , better get the down_parts mask
            # human_mask_optimazed = human_mask_detect_and_expand(loadimage_229_human_img[0], human_fashion_mask_model) # [1,2048,1536]
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
            image=get_value_at_index(loadimage_229_human_img, 0),  # human_image
        )

        # if detect clothes fashion style  and process the mask~  my router stetage
        # 按照我的思路，什么衣服，匹配什么amsk~ 来实现，也许效果有提升~
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
        box_mask = None
        for fashion_type_ in fashion_det_res[1]:
            if 'down_' in fashion_type_:   #  down_short  down_long  down_longlong
                human_mask_result2_big_low_part = human_fashion_mask_model(imageresizekj_398[0], extra_setting={'low_cover_big'})
                mask_img_pil = tensor2pil(human_mask_result2_big_low_part[1])  #  mode=L size=1242x204
                box_mask = mask_img_pil.getbbox()
                break
        if box_mask:
            box_width = box_mask[2] - box_mask[0]
            expantd_pixel = int(box_width / 10)
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
                [   tensor2pil(imageresizekj_398[0]),
                    mask_img_pil1, mask_img_pil, 
                ], cols=3, rows=1
            ).save(f'{save_dir}1_tryon_human_mask_1_judgeby_cloth_{human_id}.png')
            
        # call tryon processor
        tryon_setting = {'tops':False, 'bottoms':False, 'whole':False}
        res_image, result_dc = tryon_processor.forward(
            human_image = loadimage_229_human_img, cloth_image = loadimage_228_cloth, 
            human_mask = loadimagemask_432, req_set = tryon_setting
        )


        human_imageresize15k = imageresizekj.resize( # huaman 等效1K缩放~ 再处理~
            width=1024, height=1024, upscale_method="nearest-exact",
            keep_proportion=True, divisible_by=2, crop="disabled",
            image=get_value_at_index(loadimage_229_human_img, 0),  # human_image
        )
        human_img = tensor2pil(human_imageresize15k[0])
        human_mask = tensor2pil(loadimagemask_432[0])
        # cloth_rmbg = tensor2pil(layerutility_imageremovealpha_273_cloth_rmbg[0])
        cloth_rmbg = result_dc['cloth_rmbg']
        repaint_area_res = result_dc['middle_lq']
        refiner_img = result_dc['refiner_res']
        res_img = res_image

        endtime = time.time()
        cost_time = f'{endtime - start_time:.2f}'
        debug_image_collection = [
            human_img, 
            cloth_rmbg.resize(size=human_img.size),
            draw_text(res_img.resize(size=human_img.size), f"generated cost:{cost_time}"),
            draw_text(human_mask.resize(size=human_img.size), "human_mask"),
            # draw_text(human_mask.resize(size=human_img.size), "human_mask"),
            # draw_text(refiner_img, "refiner"),
            # res_img.resize(size = generated_raw.size)
        ]
        debug_img = make_image_grid(debug_image_collection, cols=4, rows=1).convert('RGB')
        debug_img.save(f'{save_dir}{human_id}_{inx}_res_{lego_version}.png')
        # ref.write(f"id:{human_id} _{inx}_ {lego_version} cost: {cost_time} sec\n")
        # bar.update(1)
        # break
    
if __name__ == '__main__':
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

