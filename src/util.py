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

# 不依赖任何comfyui的东西 --- 
# code tide tool 
# 工具：一般是:   ]()   实例化工具，这些对应的所有代码，部分都将放置在init之中~， 如何设计模板，才是最好的加载呢？
#   -  将所有的
# - 输入输出如何重命名，将其做到更加优雅~？

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

def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None,]


def image2mask(image: Image.Image) -> torch.Tensor:
    if isinstance(image, Image.Image):
        image = pil2tensor(image)
    return image.squeeze()[..., 0]

def mask2image(mask: torch.Tensor) -> Image.Image:
    if len(mask.shape) == 2:
        mask = mask.unsqueeze(0)
    return tensor2pil(mask)

def RGB2RGBA(image: Image.Image, mask: Union[Image.Image, torch.Tensor]) -> Image.Image:
    if isinstance(mask, torch.Tensor):
        mask = mask2image(mask)
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.LANCZOS)
    return Image.merge('RGBA', (*image.convert('RGB').split(), mask.convert('L')))


def draw_text(image, text, font_path=None, position=(50, 50), font_size=65, color=(255, 0, 0)):  # 默认白色
    draw = ImageDraw.Draw(image)
    # 根据图像模式选择适当的颜色格式
    if image.mode == 'RGB':
        color = (255, 255, 255) if color == 255 else color  # RGB模式
    elif image.mode == 'RGBA':
        color = (255, 255, 255, 255) if color == 255 else color  # RGBA模式
    elif image.mode in ['L', '1']:
        color = 255 if isinstance(color, tuple) else color  # 灰度图模式
    
    if font_path is not None:
        font = ImageFont.truetype(font_path, font_size)
    else:
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

def expand_face_box(box, width, height, expand_rate = 1.0):
    left, top, right, bottom = box
    face_w, face_h = right - left, bottom - top
    face_w_dt = face_h_dt = max(int(face_w * expand_rate) , int(face_h * expand_rate))
    center_x, center_y = left + face_w // 2, top + face_h // 2
    face_w = face_h = max(face_w, face_h)
    left, top = max(0, center_x - face_w // 2 - face_w_dt), max(0, center_y - face_h // 2 - face_h_dt)
    right, bottom = min(width, center_x + face_w // 2 + face_w_dt), min(height, center_y + face_h // 2 + face_h_dt)
    left, top, right, bottom = int(left), int(top), int(right), int(bottom)
    return (left, top, right, bottom)

def expand_bbox(bbox, image_width, image_height, expand_ratio=0.1):
    """
    扩展bbox的大小，同时确保不超出图像边界。
    
    参数:
    bbox: 列表或元组，格式为 [x_min_ratio, y_min_ratio, x_max_ratio, y_max_ratio]，为归一化的比率形式。
    image_width: 图像宽度（像素）
    image_height: 图像高度（像素）
    expand_ratio: 扩展比例，默认为0.1（10%）
    
    返回:
    expanded_bbox: 扩展后的bbox，格式与输入相同（归一化的比率形式）
    """
    assert bbox[0] <= 1.0 or bbox[1] <= 1.0 or bbox[2] <= 1.0 or bbox[3] <= 1.0
    x_min = bbox[0] * image_width
    y_min = bbox[1] * image_height
    x_max = bbox[2] * image_width
    y_max = bbox[3] * image_height
    
    width = x_max - x_min
    height = y_max - y_min
    
    width_expand = width * expand_ratio
    height_expand = height * expand_ratio
    
    new_x_min = max(0, x_min - width_expand / 2)  # 确保不小于0
    new_y_min = max(0, y_min - height_expand / 2)  # 确保不小于0
    new_x_max = min(image_width, x_max + width_expand / 2)  # 确保不超过图像宽度
    new_y_max = min(image_height, y_max + height_expand / 2)  # 确保不超过图像高度
    
    expanded_bbox = [
        new_x_min / image_width,
        new_y_min / image_height,
        new_x_max / image_width,
        new_y_max / image_height
    ]
    expanded_bbox_mx = [
        new_x_min, new_y_min,
        new_x_max, new_y_max
    ]
    
    return expanded_bbox, expanded_bbox_mx

class MaskSubtraction:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
                    "required": {
                        "masks_a": ("MASK",),
                        "masks_b": ("MASK",),
                    }
                }

    CATEGORY = "zdx/mask"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)

    FUNCTION = "subtract_masks"

    def subtract_masks(self, masks_a, masks_b):
        subtracted_masks = torch.clamp(masks_a - masks_b, 0, 255)
        return (subtracted_masks,)

class MaskAdd:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
                    "required": {
                        "masks_a": ("MASK",),
                        "masks_b": ("MASK",),
                    }
                }

    CATEGORY = "zdx/mask"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)

    FUNCTION = "add_masks"

    def add_masks(self, masks_a, masks_b):
        subtracted_masks = torch.clamp(masks_a + masks_b, 0, 255)
        return (subtracted_masks,)


# res_image = tensor2pil(imagecompositemasked_442[0])
# human_resized = tensor2pil(layerutility_imagescalebyaspectratio_v2_267[0])
# mask_resized = tensor2pil(layerutility_imagescalebyaspectratio_v2_267[0])
# cloth_rmbg = tensor2pil(layerutility_imageremovealpha_273[0])
# human_cloth_concated = tensor2pil(easy_imageconcat_275[0])
# generated_raw = tensor2pil(get_value_at_index(easy_imagessplitimage_299, 1))
# cost_time = f'{endtime - start_time:.2f}'
# debug_image_collection = [
#     human_img, 
#     draw_text(cloth_rmbg.resize(size=human_img.size),
#             human_cloeth['human_path'].split('/')[-1] ),
#     draw_text(res_image, f"generated_res cost:{cost_time}"),
#         # draw_text(human_cloth_concated, "human_cloth_concated"),
#     draw_text(generated_raw, "generated_raw"),
# ]
# debug_img = make_image_grid(debug_image_collection, cols=4, rows=1).convert('RGB')
# debug_img.save(f'{save_dir}{human_id}_{inx}_debug_{lego_version}.jpg')
# res_image.save(f'{save_dir}{human_id}_{inx}_res_{lego_version}.png')
# ref.write(f"id:{human_id} _{inx}_ {lego_version} cost: {cost_time} sec\n")



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



def extract_obj_in_box(self, image, mask):
    # 特征增强。将主体突出
    # Convert from batch format [B,C,H,W] to [C,H,W]
    img = image[0]
    mask = masks[0]
    # Find bounding box coordinates from mask
    y_indices, x_indices = torch.where(mask > 0)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return (image,)
    bbox = [
        x_indices.min().item(),
        y_indices.min().item(),
        x_indices.max().item() + 1,
        y_indices.max().item() + 1
    ]
    
    # Crop the object using bbox
    cloth_obj = img[bbox[1]:bbox[3], bbox[0]:bbox[2], : ]
    
    # Get dimensions
    h, w, _ = img.shape
    h_, w_, _ = cloth_obj.shape
    
    # Check if dimensions are valid to avoid division by zero
    if w_ == 0 or h_ == 0:
        return (image,)

    # Calculate scale to fit within original image
    scale_ = min(w/w_, h/h_)
    new_w, new_h = int(scale_ * w_), int(scale_ * h_)
    
    # Resize using interpolate
    # image_tensor = cloth_obj.permute(2, 0, 1)  # HWC ->  CHW
    cloth_obj = torch.nn.functional.interpolate(
        cloth_obj.permute(2, 0, 1).unsqueeze(0),  size=(new_h, new_w),  mode='bilinear',  align_corners=False
    ).squeeze(0).permute(1, 2, 0)
    
    # Create new blank image
    obj_expand = torch.zeros_like(img)
    
    # Calculate paste coordinates
    x = 0 if new_w == w else (w - new_w) // 2
    y = 0 if new_h == h else (h - new_h) // 2
    
    # Paste the resized object
    obj_expand[y:y+new_h, x:x+new_w, :] = cloth_obj
    # tensor2pil(obj_expand.unsqueeze(0)).save('/home/dell/study/test_comfy/img/cloth_obj_resized_tensor.jpeg')
    
    return (obj_expand.unsqueeze(0),)


def paint_bbox_tensor(mask, bbox, color=0.0):
    """
    将给定的bbox区域在PyTorch张量mask中涂抹为白色。
    参数:
    mask: torch.Tensor, 形状为 [1, H, W]，表示单通道mask。
    bbox: 列表或元组，格式为 [x_min, y_min, x_max, y_max]。
    return modified_mask: 修改后的mask。
    """
    # 提取bbox坐标
    x_min, y_min, x_max, y_max = bbox
    
    # 确保bbox坐标是整数
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
    
    # 修改mask中对应区域的值为1（白色）
    mask[:, y_min:y_max, x_min:x_max] = color
    
    return mask

def paint_bbox_pil(image, bbox, color = 0):
    """
    将给定的bbox区域在PIL图像中涂抹为白色。
    参数:
    image: PIL.Image, 输入的图像或mask。
    bbox: 列表或元组，格式为 [x_min, y_min, x_max, y_max]。
    返回:
    modified_image: 修改后的图像。
    """
    # 创建一个绘图对象
    draw = ImageDraw.Draw(image)
    
    # 提取bbox坐标
    x_min, y_min, x_max, y_max = bbox
    
    # 绘制白色矩形区域
    draw.rectangle([x_min, y_min, x_max, y_max], fill=color)
    
    return image
