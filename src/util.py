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

def draw_text(image, text, position=(50, 50), font_size=45, color=(255, 255, 255)):  # 默认白色
    draw = ImageDraw.Draw(image)
    # 根据图像模式选择适当的颜色格式
    if image.mode == 'RGB':
        color = (255, 255, 255) if color == 255 else color  # RGB模式
    elif image.mode == 'RGBA':
        color = (255, 255, 255, 255) if color == 255 else color  # RGBA模式
    elif image.mode in ['L', '1']:
        color = 255 if isinstance(color, tuple) else color  # 灰度图模式
    
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


