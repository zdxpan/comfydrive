#     ------- must add comgfyui path to sys.path -
import os
import random
import sys
from typing import Sequence, Mapping, Any, Union, Tuple
import torch
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import torch.nn as nn
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import shutil
from torchvision import transforms

from onnxruntime import InferenceSession
import onnxruntime as ort
from ultralytics import YOLO   # for human fasion detect
from huggingface_hub import hf_hub_download
import folder_paths

from src.util import (
    find_path, add_comfyui_directory_to_sys_path, add_extra_model_paths, get_value_at_index, image2mask, mask2image,
    tensor2pil, pil2tensor, pilmask2tensor, make_image_grid, draw_text, expand_face_box #, zdxApplySageAtt, #import_custom_nodes
)

from src.cloth_segment import ClothesSegment


# /data/comfy_model/RMBG/segformer_clothes   # 衣服精细分割~
# /home/dell/study/comfyui/models/yolo/deepfashion2_yolov8s-seg.pt   #  衣服分类-兼职分割~ 
# /home/dell/models/deeplabv3p-resnet50-human.onnx                   #  衣服分类- 兼职分割~


class HumanFashionMaskDetailer():
    # use down_long get much more big mask: 
    group_setting = {
        'upper_short': ['Upper-clothes', ],     # 短衣服1: Upper-clothes  Belt:腰带  Scarf 围巾~
        'upper_long': ['Upper-clothes', 'Left-arm', 'Right-arm', 'Belt'],        # 短衣服2长袖:  Upper-clothes, Left-arm, Right-arm
        'down_short':  ['Pants', 'Belt'],            # Pants,                     # 只有裤子:   
        'down_long':  ['Pants', 'Left-leg', 'Right-leg', 'Belt', 'Skirt'],              # Pants, Left-leg, Right-leg  
        'down_longlong': ['Dress', 'Upper-clothes', 'Pants', 'Left-leg', 'Right-leg', 'Skirt', 'Belt'],    # 长裙或者长裤: Pants, Left-leg, Right-leg  , Dress, skit
        'low_cover_big': ['Skirt', 'Dress', 'Belt',  'Pants',  'Left-leg', 'Right-leg', 'Left-shoe',  'Right-shoe', 'Bag'],
    }
    def __init__(self):
        self.name = self.__class__.__name__
        with torch.inference_mode():
            # self.birefnet = NODE_CLASS_MAPPINGS["BiRefNet"]()
            # self.rmbg = NODE_CLASS_MAPPINGS["RMBG"]()
            # self.clothessegment = NODE_CLASS_MAPPINGS["ClothesSegment"]()
            self.clothessegment = ClothesSegment()
            # convert_masks_to_images = NODE_CLASS_MAPPINGS["Convert Masks to Images"]()
            
    def __call__(self, input_image, extra_setting = set()):
        with torch.inference_mode():
            # self.birefnet_9 = birefnet.matting(
            #     device="auto", image=get_value_at_index(loadimage_14, 0)
            # )
            default_class_selections = {
                'Upper-clothes': True,   'Skirt': True, 'Dress': True, 'Belt': True, 
                'Pants': True,  'Left-arm': True, 'Right-arm': True, 'Left-leg': True, 'Right-leg': True,
                'Scarf': True, 'Left-shoe': True,  'Right-shoe': True, 'Bag': True,
            }

            class_selections =  {
                'Hat': False,  'Hair': False,  'Face': False, 'Sunglasses': False,   #  # 帽子  头发  脸 太阳镜 
                'Upper-clothes': False,   'Skirt': False,                           # 上衣   裙子
                'Dress': False,                              #     连衣裙
                'Belt': False,                               #     腰带
                'Pants': False,  'Left-arm': False, 'Right-arm': False,     #   长裤     左臂   右臂  
                'Left-leg': False, 'Right-leg': False,                     #   左腿     右腿 
                'Bag': False,                                             #    包
                'Scarf': False,                                           #    围巾
                'Left-shoe': False,                                       #    左鞋
                'Right-shoe': False,                                      #    右鞋
                'Background': False,                                     #      background
                'process_res': 512, 'mask_blur': 0, 'mask_offset': 0,
                'background_color': "Alpha", 'invert_output': False,
                'images': input_image,
            }
            for group_  in  extra_setting:  # set()  {'down_short', 'upper_long'}
                if group_ in HumanFashionMaskDetailer.group_setting:
                    for  set_key in HumanFashionMaskDetailer.group_setting[group_]:
                        class_selections[set_key] = True
                        print(f'>> {group_} : {set_key} seted')
            if len(extra_setting) < 1:
                class_selections.update(default_class_selections)
            class_map = {
                "Background": 0, "Hat": 1, "Hair": 2, "Sunglasses": 3, 
                "Upper-clothes": 4, "Skirt": 5, "Pants": 6, "Dress": 7,
                "Belt": 8, "Left-shoe": 9, "Right-shoe": 10, "Face": 11,
                "Left-leg": 12, "Right-leg": 13, "Left-arm": 14, "Right-arm": 15,
                "Bag": 16, "Scarf": 17
            }
            clothessegment_24 = self.clothessegment.segment_clothes(**class_selections)
            if 'low_cover_big'  in extra_setting:
                # expand the mask to minimal rect area
                pass
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


class FashionSegDetect():
    # Fashion classfy
    from ultralytics import YOLO
    fashion_names = {
        'short_sleeved_shirt' : '短袖衬衫',  'long_sleeved_shirt':'长袖衬衫',  'short_sleeved_outwear':'短袖外套',  'long_sleeved_outwear':'长袖外套',
        'vest':'背心', 'sling':'吊带', 'shorts':'短裤', 'trousers':'长裤', 'skirt':'裙子', 'short_sleeved_dress':'短袖连衣裙', 'long_sleeved_dress':'长袖连衣裙', 
        'vest_dress':'背心连衣裙', 'sling_dress':'吊带连衣裙'
    }
    fashion_label = {0: 'short_sleeved_shirt',  # '短袖衬衫'  Upper-clothes  Belt   Scarf  
    1: 'long_sleeved_shirt',        #   '长袖衬衫',     Scarf   Left-arm Right-arm
    2: 'short_sleeved_outwear',     #   '短袖外套'
    3: 'long_sleeved_outwear',      #   '长袖外套',
    4: 'vest',                      #   '背心',
    5: 'sling',                     #   '吊带',
    6: 'shorts',                    #   '短裤',
    7: 'trousers',                  #   '长裤',
    8: 'skirt',                     #   '裙子',  may be short dress?
    9: 'short_sleeved_dress',       #   '短袖连衣裙',
    10: 'long_sleeved_dress',       #   '长袖连衣裙',
    11: 'vest_dress',               #   '背心连衣裙',
    12: 'sling_dress'}              #   '吊带连衣裙'
    fashion_label_index_dc = {v:k for k,v in fashion_label.items()}
    group_cls = {
        'upper_short': ['short_sleeved_shirt', 'short_sleeved_outwear', 'vest', 'sling', 'short_sleeved_dress'],     # 短衣服1: Upper-clothes  Belt:腰带  Scarf 围巾~
        'upper_long': ['long_sleeved_shirt', 'long_sleeved_outwear',],        # 短衣服2长袖:  Upper-clothes, Left-arm, Right-arm
        'down_short':  ['shorts', ],            # Pants,                     # 只有裤子:   
        'down_long':  ['trousers', 'skirt', ],              # Pants, Left-leg, Right-leg  
        'down_longlong': ['trousers', 'long_sleeved_dress', 'vest_dress', 'sling_dress', 'skirt', 'short_sleeved_dress'],    # 长裙或者长裤: Pants, Left-leg, Right-leg  , Dress, skit
    }
    class_group_dc = {sub_v: k for k, v in group_cls.items() for sub_v in v}

    def __init__(self, model_path):
        # deeplabv3p-resnet50-human.onnx'
        # deepfashion2_yolov8s-seg.pt'
        self.model = YOLO(model_path)
    
    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            image = tensor2pil(image)
        output = self.model(image)
        # pred = output[0].plot()         # for visualize
        # pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        # pred = Image.fromarray(pred)
        # pred.resize(size = (256, 512))
        if not output or len(output) < 1:
            return []
        res = []
        res_cn_name = []
        group_setting = set()
        for label_index in output[0].boxes.cls.cpu().numpy():        #   output[0].boxes.cls, output[0].boxes.conf
            label_name = FashionSegDetect.fashion_label[label_index]
            group = FashionSegDetect.class_group_dc[label_name]
            cn_name = FashionSegDetect.fashion_names[label_name]
            res_cn_name.append(cn_name)
            group_setting.add(group)
            res.append(label_name)
        return res, group_setting, res_cn_name

# add huma parse segment  model node~
# 人类分割器 : 实测，精度非常低。只适合用于检测~
class HumanSegmentParts:
    """
    This node is used to get a mask of the human parts in the image.

    The model used is DeepLabV3+ with a ResNet50 backbone trained
    by Keras-io, converted to ONNX format.

    """
    labels = {
        0: ("background", "Background"),
        1: ("hat", "Hat: Hat, helmet, cap, hood, veil, headscarf, part covering the skull and hair of a hood/balaclava, crown…",),
        2: ("hair", "Hair", ),
        3: ("glove", "Glove",),
        4: ("glasses","Sunglasses/Glasses: Sunglasses, eyewear, protective glasses…",),
        5: ("upper_clothes","UpperClothes: T-shirt, shirt, tank top, sweater under a coat, top of a dress…",),
        6: ("face_mask","Face Mask: Protective mask, surgical mask, carnival mask, facial part of a balaclava, visor of a helmet…",),
        7: ("coat","Coat: Coat, jacket worn without anything on it, vest with nothing on it, a sweater with nothing on it…",),
        8: ("socks","Socks",),
        9: ("pants","Pants: Pants, shorts, tights, leggings, swimsuit bottoms… (clothing with 2 legs)",),
        10: ("torso-skin", "Torso-skin",),
        11: ("scarf","Scarf: Scarf, bow tie, tie…",),
        12: ("skirt","Skirt: Skirt, kilt, bottom of a dress…",),
        13: ("face","Face",),
        14: ("left-arm","Left-arm (naked part)",),
        15: ("right-arm","Right-arm (naked part)",),
        16: ("left-leg","Left-leg (naked part)",),
        17: ("right-leg","Right-leg (naked part)",),
        18: ("left-shoe","Left-shoe",),
        19: ("right-shoe", "Right-shoe", ),
        20: ("bag", "Bag: Backpack, shoulder bag, fanny pack… (bag carried on oneself",),
        21: ("","Others: Jewelry, tags, bibs, belts, ribbons, pins, head decorations, headphones…",),
    }
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "get_mask"
    CATEGORY = "Metal3d"
    OUTPU_NODE = True

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def get_mask(self, image: torch.Tensor, **kwargs) -> Tuple[torch.Tensor]:
        """
        Return a Tensor with the mask of the human parts in the image.
        """
        if self.model is None:
            self.model = ort.InferenceSession(self.model_path)
        ret_tensor, class_label_score = self.get_human_mask(image, model=self.model, rotation=0, **kwargs)

        return (ret_tensor, class_label_score, )

    @staticmethod
    def get_human_mask(
        image: torch.Tensor, model: InferenceSession, rotation: float, **kwargs
    ) -> Tuple[torch.Tensor, int]:
        """
        Return a Tensor with the mask of the human parts in the image.

        The rotation parameter is not used for now. The idea is to propose rotation to help
        the model to detect the human parts in the image if the character is not in a casual position.
        Several tests have been done, but the model seems to fail to detect the human parts in these cases,
        and the rotation does not help.
        """

        image = image.squeeze(0)
        image_np = image.numpy() * 255

        pil_image = Image.fromarray(image_np.astype(np.uint8))
        original_size = pil_image.size  # to resize the mask later
        # resize to 512x512 as the model expects
        pil_image = pil_image.resize((512, 512))
        center = (256, 256)

        if rotation != 0:
            pil_image = pil_image.rotate(rotation, center=center)

        # normalize the image
        image_np = np.array(pil_image).astype(np.float32) / 127.5 - 1
        image_np = np.expand_dims(image_np, axis=0)

        # use the onnx model to get the mask
        input_name = model.get_inputs()[0].name
        output_name = model.get_outputs()[0].name
        result = model.run([output_name], {input_name: image_np})
        result = np.array(result[0]).argmax(axis=3).squeeze(0)

        score: int = 0

        mask = np.zeros_like(result)

        # Initialize result dictionary with all parts set to False
        detect_label_dc = {part_name: 0 for part_name, _ in HumanSegmentParts.labels.values() if part_name}
    
        for index, label_tuple in HumanSegmentParts.labels.items():
            class_name = label_tuple[0]
            class_index = index
            # for class_name, enabled in kwargs.items():
            # class_index = HumanSegmentParts.get_class_index(class_name)
            index_area = (result == class_index).sum()
            index_area = index_area / result.size
            if index_area > 0:
                detect_label_dc[class_name] = index_area
                mask[result == class_index] = 50 + class_index * 10
                # score += mask.sum()
        
        sorted_items = sorted(detect_label_dc.items(), key=lambda item: item[1], reverse=True)
        sorted_dict = dict(sorted_items)
        # back to the original size
        mask_image = Image.fromarray(mask.astype(np.uint8), mode="L")
        if rotation != 0:
            mask_image = mask_image.rotate(-rotation, center=center)

        mask_image = mask_image.resize(original_size)

        # and back to numpy...
        mask = np.array(mask_image).astype(np.float32) / 255

        # add 2 dimensions to match the expected output
        mask = np.expand_dims(mask, axis=0)
        mask = np.expand_dims(mask, axis=0)
        # ensure to return a "binary mask_image"

        del image_np, result  # free up memory, maybe not necessary
        return (torch.from_numpy(mask.astype(np.uint8)), sorted_dict)

