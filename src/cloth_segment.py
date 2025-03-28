#  --- 
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

from huggingface_hub import hf_hub_download
import folder_paths


def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None,]

def tensor2pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8))

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

device = "cuda" if torch.cuda.is_available() else "cpu"

folder_paths.add_model_folder_path("rmbg", os.path.join(folder_paths.models_dir, "RMBG"))

AVAILABLE_MODELS = {
    "segformer_b2_clothes": "1038lab/segformer_clothes"
}

class ClothesSegment:
    def __init__(self, model_path=None):
        self.processor = None
        self.model = None
        self.cache_dir = os.path.join(folder_paths.models_dir, "RMBG", "segformer_clothes")
    
    @classmethod
    def INPUT_TYPES(cls):
        available_classes = ["Hat", "Hair", "Face", "Sunglasses", "Upper-clothes", "Skirt", "Dress", "Belt", "Pants", "Left-arm", "Right-arm", "Left-leg", "Right-leg", "Bag", "Scarf", "Left-shoe", "Right-shoe","Background"]
        
        tooltips = {
            "process_res": "Processing resolution (higher = more VRAM)",
            "mask_blur": "Blur amount for mask edges",
            "mask_offset": "Expand/Shrink mask boundary",
            "background_color": "Choose background color (Alpha = transparent)",
            "invert_output": "Invert both image and mask output",
        }
        
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                **{cls_name: ("BOOLEAN", {"default": False}) 
                   for cls_name in available_classes},
                "process_res": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 32, "tooltip": tooltips["process_res"]}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "tooltip": tooltips["mask_blur"]}),
                "mask_offset": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1, "tooltip": tooltips["mask_offset"]}),
                "background_color": (["Alpha", "black", "white", "gray", "green", "blue", "red"], {"default": "Alpha", "tooltip": tooltips["background_color"]}),
                "invert_output": ("BOOLEAN", {"default": False, "tooltip": tooltips["invert_output"]}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK")
    FUNCTION = "segment_clothes"
    CATEGORY = "🧪AILab/🧽RMBG"

    def check_model_cache(self):
        if not os.path.exists(self.cache_dir):
            return False, "Model directory not found"
        
        required_files = [
            'config.json',
            'model.safetensors',
            'preprocessor_config.json'
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(self.cache_dir, f))]
        if missing_files:
            return False, f"Required model files missing: {', '.join(missing_files)}"
        return True, "Model cache verified"

    def clear_model(self):
        if self.model is not None:
            self.model.cpu()
            del self.model
            self.model = None
            self.processor = None
            torch.cuda.empty_cache()

    def download_model_files(self):
        model_id = AVAILABLE_MODELS["segformer_b2_clothes"]
        model_files = {
            'config.json': 'config.json',
            'model.safetensors': 'model.safetensors',
            'preprocessor_config.json': 'preprocessor_config.json'
        }
        
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"Downloading Clothes Segformer model files...")
        
        try:
            for save_name, repo_path in model_files.items():
                print(f"Downloading {save_name}...")
                downloaded_path = hf_hub_download(
                    repo_id=model_id,
                    filename=repo_path,
                    local_dir=self.cache_dir,
                    local_dir_use_symlinks=False
                )
                
                if os.path.dirname(downloaded_path) != self.cache_dir:
                    target_path = os.path.join(self.cache_dir, save_name)
                    shutil.move(downloaded_path, target_path)
            return True, "Model files downloaded successfully"
        except Exception as e:
            return False, f"Error downloading model files: {str(e)}"

    def segment_clothes(self, images, process_res=1024, mask_blur=0, mask_offset=0, background_color="Alpha", invert_output=False, **class_selections):
        try:
            # Check and download model if needed
            cache_status, message = self.check_model_cache()
            if not cache_status:
                print(f"Cache check: {message}")
                download_status, download_message = self.download_model_files()
                if not download_status:
                    raise RuntimeError(download_message)
            
            # Load model if needed
            if self.processor is None:
                self.processor = SegformerImageProcessor.from_pretrained(self.cache_dir)
                self.model = AutoModelForSemanticSegmentation.from_pretrained(self.cache_dir)
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False
                self.model.to(device)

            # Class mapping for segmentation
            class_map = {
                "Background": 0, "Hat": 1, "Hair": 2, "Sunglasses": 3, 
                "Upper-clothes": 4, "Skirt": 5, "Pants": 6, "Dress": 7,
                "Belt": 8, "Left-shoe": 9, "Right-shoe": 10, "Face": 11,
                "Left-leg": 12, "Right-leg": 13, "Left-arm": 14, "Right-arm": 15,
                "Bag": 16, "Scarf": 17
            }

            # Get selected classes
            selected_classes = [name for name, selected in class_selections.items() if selected]
            if not selected_classes:
                selected_classes = ["Upper-clothes"]

            # Image preprocessing
            transform_image = transforms.Compose([
                transforms.Resize((process_res, process_res)),
                transforms.ToTensor(),
            ])

            batch_tensor = []
            batch_masks = []
            
            for image in images:
                orig_image = tensor2pil(image)
                w, h = orig_image.size
                
                input_tensor = transform_image(orig_image)

                if input_tensor.shape[0] == 4:
                    input_tensor = input_tensor[:3]
                
                input_tensor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(input_tensor)
                
                input_tensor = input_tensor.unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    logits = outputs.logits.cpu()
                    upsampled_logits = nn.functional.interpolate(
                        logits,
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    pred_seg = upsampled_logits.argmax(dim=1)[0]

                    # Combine selected class masks
                    combined_mask = None
                    for class_name in selected_classes:
                        mask = (pred_seg == class_map[class_name]).float()
                        if combined_mask is None:
                            combined_mask = mask
                        else:
                            combined_mask = torch.clamp(combined_mask + mask, 0, 1)

                    # Convert mask to PIL for processing
                    mask_image = Image.fromarray((combined_mask.numpy() * 255).astype(np.uint8))

                    if mask_blur > 0:
                        mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=mask_blur))

                    if mask_offset != 0:
                        if mask_offset > 0:
                            mask_image = mask_image.filter(ImageFilter.MaxFilter(size=mask_offset * 2 + 1))
                        else:
                            mask_image = mask_image.filter(ImageFilter.MinFilter(size=-mask_offset * 2 + 1))

                    if invert_output:
                        mask_image = Image.fromarray(255 - np.array(mask_image))

                    # Handle background color
                    if background_color == "Alpha":
                        rgba_image = RGB2RGBA(orig_image, mask_image)
                        result_image = pil2tensor(rgba_image)
                    else:
                        bg_colors = {
                            "black": (0, 0, 0),
                            "white": (255, 255, 255),
                            "gray": (128, 128, 128),
                            "green": (0, 255, 0),
                            "blue": (0, 0, 255),
                            "red": (255, 0, 0)
                        }
                        
                        rgba_image = RGB2RGBA(orig_image, mask_image)
                        bg_image = Image.new('RGBA', orig_image.size, (*bg_colors[background_color], 255))
                        composite_image = Image.alpha_composite(bg_image, rgba_image)
                        result_image = pil2tensor(composite_image.convert('RGB'))

                    batch_tensor.append(result_image)
                    batch_masks.append(pil2tensor(mask_image))

            # Prepare final output
            batch_tensor = torch.cat(batch_tensor, dim=0)
            batch_masks = torch.cat(batch_masks, dim=0)
            
            return (batch_tensor, batch_masks)

        except Exception as e:
            self.clear_model()
            raise RuntimeError(f"Error in Clothes Segformer processing: {str(e)}")
        finally:

            if not self.model.training:
                self.clear_model()

