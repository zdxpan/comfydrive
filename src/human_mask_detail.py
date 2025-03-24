import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
from PIL import Image, ImageDraw


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





from src.util import (
    find_path, add_comfyui_directory_to_sys_path, add_extra_model_paths, get_value_at_index, image2mask, mask2image,
    tensor2pil, pil2tensor, pilmask2tensor, make_image_grid, draw_text #, zdxApplySageAtt, #import_custom_nodes
)


def whiten_box(image, box):
    """
    在给定的 box 区域内将图像涂抹为白色。
    """
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, fill="white")
    return image



class HumanMaskDetailer():
    def __init__(self):
        self.name = self.__class__.__name__
        with torch.inference_mode():
            # self.birefnet = NODE_CLASS_MAPPINGS["BiRefNet"]()
            # self.rmbg = NODE_CLASS_MAPPINGS["RMBG"]()
            self.clothessegment = NODE_CLASS_MAPPINGS["ClothesSegment"]()
            self.masktoimage = NODE_CLASS_MAPPINGS["MaskToImage"]()
            # convert_masks_to_images = NODE_CLASS_MAPPINGS["Convert Masks to Images"]()
            
    def __call__(self, input_image):
        with torch.inference_mode():
            # self.birefnet_9 = birefnet.matting(
            #     device="auto", image=get_value_at_index(loadimage_14, 0)
            # )
            class_selections =  {
                    'Hat': False,  'Hair': False,  'Face': False, 'Sunglasses': False,
                    'Upper-clothes': True,   'Skirt': True,
                    'Dress': True,
                    'Belt': True, 
                    'Pants': True,  'Left-arm': True, 'Right-arm': True,
                    'Left-leg': True, 'Right-leg': True,
                    'Bag': True, 
                    'Scarf': True,
                    'Left-shoe': True,
                    'Right-shoe': True,
                    'Background': False, 
                    'process_res': 512, 'mask_blur': 0, 'mask_offset': 0,
                    'background_color': "Alpha", 'invert_output': False,
                    'images': get_value_at_index(loadimage_14, 0),
                }
            class_map = {
                "Background": 0, "Hat": 1, "Hair": 2, "Sunglasses": 3, 
                "Upper-clothes": 4, "Skirt": 5, "Pants": 6, "Dress": 7,
                "Belt": 8, "Left-shoe": 9, "Right-shoe": 10, "Face": 11,
                "Left-leg": 12, "Right-leg": 13, "Left-arm": 14, "Right-arm": 15,
                "Bag": 16, "Scarf": 17
            }
            clothessegment_24 = self.clothessegment.segment_clothes(**class_selections)
        return clothessegment_24

# human_masker.clothessegment.cache_dir
# 'CATEGORY', 'FUNCTION', 'INPUT_TYPES', 'RELATIVE_PYTHON_MODULE', 'RETURN_NAMES', 'RETURN_TYPES', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__',
#  'cache_dir', 'check_model_cache', 'clear_model', 'download_model_files', 'model', 'processor', 'segment_clothes']
# cache_dir: /data/comfy_model/RMBG/segformer_clothes 
# check_model_cache human_masker.clothessegment.check_model_cache  # (True, 'Model cache verified')
# human_masker.clothessegment.model  None


class ObjExtractByMask:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
                    "required": {
                        "image": ("IMAGE",), "masks": ("MASK",),
                    }
                }
    CATEGORY = "zdx/mask"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)

    FUNCTION = "resize_obj_area_to_max_size"

    # def extract(self, image, masks):
    def resize_obj_area_to_max_size(self, image, mask):
        # TODO 寻找最小外接矩形，多边形      # 保持比例等比缩放算法~
        image_pil = tensor2pil(image)
        mask_pil = mask2image(mask)
            
        bbox = mask_pil.getbbox()        #  最小外接矩形, 未带~ 扩展缩放~
        cloth_obj = image_pil.crop(bbox)
        w,h = image_pil.size   #  # 原始图像的宽高
        w_,h_ = cloth_obj.size    #  # 目标区域
        scale_ = min(w/w_ , h/ h_)
        new_w,new_h = int(scale_ * w_), int(scale_ * h_)
        cloth_obj = cloth_obj.resize(size = (new_w,new_h))    # # 缩放目标区域
        # pasted back to the orignal size~
        obj_expand = Image.new(cloth_obj.mode, (w,h), (0, 0, 0))
        x = 0 if new_w == w else (w - new_w) // 2 
        y = 0 if new_h == h else (h - new_h) // 2
        obj_expand.paste(cloth_obj, (x,y))
        # obj_expand.save('/home/dell/study/test_comfy/img/cloth_obj_resized_inpil.jpeg')
        return (pil2tensor(obj_expand), )



if __name__ == '__main__':
    # from nodes import NODE_CLASS_MAPPINGS
    import_custom_nodes()
    human_masker = HumanMaskDetailer()
    loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
    masktoimage = NODE_CLASS_MAPPINGS["MaskToImage"]()
    loadimage_14 = loadimage.load_image(image="image (1).png")
    load_pil_img = tensor2pil(loadimage_14[0])  #   (1242, 2048)
    human_mask_result = human_masker(loadimage_14)
    seg_res = tensor2pil(human_mask_result[0])   # RGBA
    mask_img = mask2image(human_mask_result[1])  #  mode=L size=1242x2048
    obj_extrat = ObjExtractByMask()
    cloth_obj_expand_by_node = obj_extrat.resize_obj_area_to_max_size(loadimage_14[0], human_mask_result[1])

    # TODO 寻找最小外接矩形，多边形      # 保持比例等比缩放算法~
    bbox = mask_img.getbbox()
    mask_box_white_img = whiten_box(mask_img, bbox)  # just for human mask ,not use for obj_area
    cloth_obj_expand = resize_obj_area_to_max_size(load_pil_img, mask_img)
    cloth_obj_expand.save('/home/dell/study/test_comfy/img/cloth_obj_resized.jpeg')
    
    
    # resize the box area back to orignal size, for better cloth replace
    convert_masks_to_images_16 = masktoimage.mask_to_image(human_mask_result[1])  # tensor [1, 2048, 1242, 3]

    debug_img = make_image_grid(
        [
            tensor2pil(loadimage_14[0]),
            tensor2pil(human_mask_result[0]),
        ], cols=2, rows=1).convert('RGB')
    debug_img.save(f'{save_dir}/1_refiner_debug_{lego_version}.jpg')

