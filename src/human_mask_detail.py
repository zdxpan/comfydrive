import os
import random
import sys
from typing import Sequence, Mapping, Any, Union, Tuple
import torch
from PIL import Image, ImageDraw, ImageChops
import numpy as np
import torch
from torch.nn import functional as F
from onnxruntime import InferenceSession
import onnxruntime as ort
from ultralytics import YOLO   # for human fasion detect

# 模型路径加载~  comfyui.util.folder_paths

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
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





from src.util import (
    find_path, add_comfyui_directory_to_sys_path, add_extra_model_paths, get_value_at_index, image2mask, mask2image,
    tensor2pil, pil2tensor, pilmask2tensor, make_image_grid, draw_text, expand_face_box, #, zdxApplySageAtt, #import_custom_nodes
    MaskSubtraction, MaskAdd
)
from src.mask_seg_det import HumanFashionMaskDetailer, HumanSegmentParts, FashionSegDetect, ObjExtractByMask



def whiten_box(image, box):
    """
    在给定的 box 区域内将图像涂抹为白色。
    """
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, fill="white")
    return image




def horizontal_expand(mask: torch.Tensor, expand: int) -> torch.Tensor:
    """
    仅横向(水平方向)膨胀mask
    :param mask: 输入mask张量 (H,W)或(B,H,W)
    :param expand: 膨胀像素数 (必须>=0)
    :return: 膨胀后的mask
    """
    if expand == 0:
        return mask
    
    # 确保输入是3D张量 (B,H,W)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    
    # 创建1D水平卷积核 [1,1,1,..1] (长度为2*expand+1)
    kernel_size = 2 * expand + 1
    kernel = torch.ones((1, 1, 1, kernel_size), 
                       dtype=torch.float32, 
                       device=mask.device)
    
    # 使用卷积实现水平膨胀
    # 输入形状: (B, 1, H, W)
    mask = mask.unsqueeze(1)  # 添加通道维度
    padding = (0, expand)  # 只在宽度方向填充
    
    # 反射填充边界更合理
    padded = F.pad(mask, (expand, expand, 0, 0), mode='reflect')
    
    # 使用最大值滤波模拟膨胀
    # 使用groups参数实现批量处理
    expanded = F.conv2d(padded, kernel, 
                       padding=(0, 0),  # 已经手动填充
                       groups=mask.size(0))
    
    # 阈值化并保持原始值范围
    result = (expanded > 0).float() * mask.max()
    
    return (result.squeeze(1), )  # 移除通道维度

class HumanMaskSegDetTool():
    @staticmethod
    def human_mask_detect_and_expand_with_setting(human_img, human_fashion_mask_model, resize_2k = False, setting={}):
        '''input  tensor of human or clothe image  get mask and expand the down_body Add whole mask
        ~等效缩放到2K处理，优化性能~
        usage: human_mask_img = human_mask_detect_and_expand(loadimage_229[0], human_fashion_mask_model)
        return (mask,)
        '''
        
        _, h_, w_, c_ = human_img.shape
        resized = False
        normalize_image = human_img
        if resize_2k and (h_ > 2048 or w_ > 2048):
            normalize_image = imageresizekj.resize(
                width=2048, height=2048, upscale_method="nearest-exact", keep_proportion=True,
                divisible_by=2, crop="disabled",
                image=human_img,  # human_image
            )[0]  #  resized_im, resize_w, resiz_h
            resized = True
        mask_img = human_fashion_mask_model(normalize_image, extra_setting=setting)  # 全身
        mask_img2 = human_fashion_mask_model(normalize_image, extra_setting={'low_cover_big'})  # 下半身
        mask_img_pil2 = mask2image(mask_img2[1])
        mask_bbox_2 = mask_img_pil2.getbbox()
        res_mask = mask_img[1]   # [1, 2048, 1536])
        if mask_bbox_2:
            # expand_mask_and_fill_white
            # w, h = mask_img_pil2.size
            # mask_bbox_2 = expand_face_box(mask_bbox_2, w, h, expand_rate=0.5)
            # test_expand_mask GrowMask
            human_mask_growed3 = horizontal_expand(mask_img2[1], 120)[0]  # 性能最高~
            human_mask_add = MaskAdd().add_masks(human_mask_growed3[0], mask_img[1])[0]
            res_mask = human_mask_add
        if resized:
            res_mask = resizemask.resize(
                width=w_, height=h_, keep_proportions=False, upscale_method="nearest-exact",
                crop="disabled", mask=res_mask,
            )[0]
        return (res_mask, )

    @staticmethod
    def human_mask_detect_and_expand(human_img, human_fashion_mask_model, resize_2k = False):
        '''input  tensor of human or clothe image  get mask and expand the down_body Add whole mask
        ~等效缩放到2K处理，优化性能~
        usage: human_mask_img = human_mask_detect_and_expand(loadimage_229[0], human_fashion_mask_model)
        return (mask,)
        '''
        
        _, h_, w_, c_ = human_img.shape
        resized = False
        normalize_image = human_img
        if resize_2k and (h_ > 2048 or w_ > 2048):
            normalize_image = imageresizekj.resize(
                width=2048, height=2048, upscale_method="nearest-exact", keep_proportion=True,
                divisible_by=2, crop="disabled",
                image=human_img,  # human_image
            )[0]  #  resized_im, resize_w, resiz_h
            resized = True
        mask_img = human_fashion_mask_model(normalize_image)  # 全身
        mask_img2 = human_fashion_mask_model(normalize_image, extra_setting={'low_cover_big'})  # 下半身
        mask_img_pil2 = mask2image(mask_img2[1])
        mask_bbox_2 = mask_img_pil2.getbbox()
        res_mask = mask_img[1]   # [1, 2048, 1536])
        if mask_bbox_2:
            # expand_mask_and_fill_white
            # w, h = mask_img_pil2.size
            # mask_bbox_2 = expand_face_box(mask_bbox_2, w, h, expand_rate=0.5)
            # test_expand_mask GrowMask
            human_mask_growed3 = horizontal_expand(mask_img2[1], 120)[0]  # 性能最高~
            human_mask_add = MaskAdd().add_masks(human_mask_growed3[0], mask_img[1])[0]
            res_mask = human_mask_add
        if resized:
            res_mask = resizemask.resize(
                width=w_, height=h_, keep_proportions=False, upscale_method="nearest-exact",
                crop="disabled", mask=res_mask,
            )[0]
        return (res_mask, )
        # for debug~
        human_img_pil = tensor2pil(human_img)
        human_mask_image = mask2image(mask_img[1])     # 全身对应的mask 
        mask_img_expand = mask2image(human_mask_growed3)
        debug_img = make_image_grid(
            [
                human_img_pil,
                draw_text(mask_img_pil2.resize(size=human_img_pil.size), "mask_ lowe_big"), 
                draw_text(mask_img_expand.resize(size=human_img_pil.size), "mask_ lowe_big expanded"), 
                draw_text(human_mask_image.resize(size=human_img_pil.size), "mask_ all body _mask"), 
                draw_text(
                    tensor2pil( res_mask).resize(size=human_img_pil.size), "mask_ added"
                ), 
                # draw_text(   # add mask in pil mode
                #     ImageChops.add(mask_img_expand.resize(size=human_img_pil.size), human_mask_image.convert("L").resize(size=human_img_pil.size)),
                #     "mask_added"
                # )
            ], cols=5, rows=1
        )
        debug_img.save('/home/dell/study/test_comfy/img/1_debug_human_im_mask_all_lowWhole.jpg')



# 本次算法优化 依赖的模型安装：
# /data/comfy_model/RMBG/segformer_clothes   # 衣服精细分割~
# /home/dell/study/comfyui/models/yolo/deepfashion2_yolov8s-seg.pt   #  衣服分类-兼职分割~ 
# /home/dell/models/deeplabv3p-resnet50-human.onnx                   #  衣服分类- 兼职分割~


if __name__ == '__main__':
    import glob
    from diffusers.utils import make_image_grid
    font_path = '/home/dell/study/test_comfy/wqy-microhei.ttc'
    # from nodes import NODE_CLASS_MAPPINGS
    import_custom_nodes()
    loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
    masktoimage = NODE_CLASS_MAPPINGS["MaskToImage"]()

    # fashion_image & fashion detect 2~
    fashion_images = glob.glob('/home/dell/study/test_comfy/img/human_mask/cloth_*.jpg')
    human_images = glob.glob('/home/dell/study/test_comfy/img/human_mask/image_*.jpeg')
    human_orig_mask = {k: k.replace('image_', 'mask_').replace('jpeg', 'png') for k in human_images}
    fashion_cls_model = '/home/dell/study/comfyui/models/yolo/deepfashion2_yolov8s-seg.pt'
    human_fashion_mask_model = HumanFashionMaskDetailer()
    fashion_detect_model =  FashionSegDetect(fashion_cls_model)
    fashion_det_res = fashion_detect_model(fashion_images[0])  #  in:PIL or path  (['long_sleeved_shirt', 'shorts'], {'down_short', 'upper_long'})
    fashion_images = []  # skip 
    for _, image_path in enumerate(fashion_images):
        cloth_image = Image.open(image_path)
        fashion_det_res = fashion_detect_model(cloth_image)
        label_texts = [i[0]+'_'+i[1]  for i in zip(fashion_det_res[0], fashion_det_res[2])]
        label_text = '\n'.join(label_texts)
        draw_text(cloth_image, label_text, font_path=font_path)
        file_name = image_path.split('/')[-1]
        # cloth_image.convert("RGB").save(f'/home/dell/study/test_comfy/img/fashion_classfy/{file_name}')
        #  假设 使用这种类型的衣服的时候，让每个人都试穿一下，那么每个人对应的mask应该处理的样式如下~
        for human_inx_, (human_, human_orig_mask_) in enumerate(human_orig_mask.items()):
            human_im_ = Image.open(human_)
            human_orig_mask_im_ = Image.open(human_orig_mask_)
            
            human_mask_result2 = human_fashion_mask_model(pil2tensor(human_im_), extra_setting=fashion_det_res[1])
            seg_res = tensor2pil(human_mask_result2[0])   # RGBA
            mask_img = mask2image(human_mask_result2[1])  #  mode=L size=1242x204
            debug_img = make_image_grid(
                [
                    cloth_image, human_im_.resize(size=cloth_image.size), 
                    seg_res.resize(size=cloth_image.size), 
                    draw_text(mask_img.resize(size=cloth_image.size), "mask_ By clothe choosed"), 
                    draw_text(human_orig_mask_im_.resize(size=cloth_image.size), "original_mask"),
                    draw_text(
                        ImageChops.add(mask_img.resize(size=cloth_image.size), human_orig_mask_im_.convert("L").resize(size=cloth_image.size)),
                        "mask_added"
                    )
                ], rows=1, cols=6
            )
            debug_img.save(f'/home/dell/study/test_comfy/img/fashion_classfy/debug_{file_name.replace(".jpg", "")}_{human_inx_}.jpg')
        # 判断面积，如果本来面积挺大，那就不改了？  避免此处的分割模型，做的不好，识别不到位~
        # 判断存在非联通区域，使用膨胀后的大大大mask,期望效果更好~

    if 1:  # test if human rembg with big low_parts :
        for human_inx_, (human_, human_orig_mask_) in enumerate(human_orig_mask.items()):
            file_name = human_.split('/')[-1]
            human_im_ = Image.open(human_)
            human_orig_mask_im_ = Image.open(human_orig_mask_)
            
            human_mask_result2 = human_fashion_mask_model(pil2tensor(human_im_), extra_setting={'low_cover_big'})
            seg_res = tensor2pil(human_mask_result2[0])   # RGBA
            mask_img = mask2image(human_mask_result2[1])  #  mode=L size=1242x204
            debug_img = make_image_grid(
                [
                    human_im_, 
                    seg_res.resize(size=human_im_.size), 
                    draw_text(mask_img.resize(size=human_im_.size), "mask_ By clothe choosed"), 
                    draw_text(human_orig_mask_im_.resize(size=human_im_.size), "original_mask"),
                    draw_text(
                        ImageChops.add(mask_img.resize(size=human_im_.size), human_orig_mask_im_.convert("L").resize(size=human_im_.size)),
                        "mask_added"
                    )
                ], rows=1, cols=5
            )
            debug_img.save(f'/home/dell/study/test_comfy/img/fashion_classfy/debug_{file_name.replace(".jpg", "")}_{human_inx_}.jpg')
    # human _ image 
    loadimage_14 = loadimage.load_image(image="image (1).png")
    load_pil_img = tensor2pil(loadimage_14[0])  #   (1242, 2048)
    # 精细人体衣服分割 default 支持传输extra_setting 控制分割上衣还是裤子，还是裙子还是全身衣服~
    human_fashion_mask_model = HumanFashionMaskDetailer()
    human_mask_result = human_fashion_mask_model(loadimage_14[0])
    human_mask_result2 = human_fashion_mask_model(loadimage_14[0], extra_setting=fashion_det_res[1])
    
    seg_res = tensor2pil(human_mask_result[0])   # RGBA
    mask_img = mask2image(human_mask_result[1])  #  mode=L size=1242x2048
    
    # 衣服特征增强算法~ 保持比例等比缩放~
    obj_extrat = ObjExtractByMask()  # 衣服特征增强算法~ 保持比例等比缩放
    cloth_obj_expand_by_node = obj_extrat.resize_obj_area_to_max_size(loadimage_14[0], human_mask_result[1])
    cloth_obj_expand = tensor2pil(cloth_obj_expand_by_node[0])
    cloth_obj_expand.save('/home/dell/study/test_comfy/img/cloth_obj_resized.jpeg')
    
    # 人体检测 1 - 衣服类型区分
    model_path='/home/dell/models/deeplabv3p-resnet50-human.onnx'
    # human_parts = HumanSegmentParts(model_path='/home/dell/models/deeplabv3p-resnet50-human.onnx')
    # # 调用 get_mask 方法获取人体部位掩码  # 启用你想要检测的部位    
    # image_tensor = loadimage_14[0]
    # human_seg_mask, human_seg_cls = human_parts.get_mask(image=image_tensor,)

    # human_mask_img = mask2image(human_seg_mask[0])
    # human_mask_img.save('/home/dell/study/test_comfy/img/1_human_segmen_mask.jpeg')



    # resize the box area back to orignal size, for better cloth replace
    convert_masks_to_images_16 = masktoimage.mask_to_image(human_mask_result[1])  # tensor [1, 2048, 1242, 3]

    debug_img = make_image_grid(
        [
            tensor2pil(loadimage_14[0]),
            tensor2pil(human_mask_result[0]),
        ], cols=2, rows=1).convert('RGB')
    # debug_img.save(f'{save_dir}/1_refiner_debug_{lego_version}.jpg')

