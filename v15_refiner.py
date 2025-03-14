import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
from diffusers.utils import make_image_grid

from src.util import (
    tensor2pil, pil2tensor, pilmask2tensor, 
)

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


def main():
    import_custom_nodes()
    with torch.inference_mode():
        alignyourstepsscheduler = NODE_CLASS_MAPPINGS["AlignYourStepsScheduler"]()
        alignyourstepsscheduler_1 = alignyourstepsscheduler.get_sigmas(
            model_type="SD1", steps=18, denoise=0.5
        )

        ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        ksamplerselect_2 = ksamplerselect.get_sampler(sampler_name="dpmpp_3m_sde_gpu")

        controlnetloader = NODE_CLASS_MAPPINGS["ControlNetLoader"]()
        controlnetloader_3 = controlnetloader.load_controlnet(
            control_net_name="control_v11f1e_sd15_tile.pth"
        )

        checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
        checkpointloadersimple_7 = checkpointloadersimple.load_checkpoint(
            ckpt_name="juggernaut_reborn.safetensors"
        )

        loraloader = NODE_CLASS_MAPPINGS["LoraLoader"]()
        loraloader_8 = loraloader.load_lora(
            lora_name="more_details.safetensors",
            strength_model=0.25,
            strength_clip=0.25,
            model=get_value_at_index(checkpointloadersimple_7, 0),
            clip=get_value_at_index(checkpointloadersimple_7, 1),
        )

        loraloader_9 = loraloader.load_lora(
            lora_name="SDXLrender_v2.0.safetensors",
            strength_model=0.1,
            strength_clip=0.1,
            model=get_value_at_index(loraloader_8, 0),
            clip=get_value_at_index(loraloader_8, 1),
        )

        upscalemodelloader = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
        upscalemodelloader_14 = upscalemodelloader.load_model(
            model_name="4xnomosunidatBokeh_v20.safetensors"
        )

        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_16 = loadimage.load_image(image="2.jpg")
        # skiped the upscale mode
        imagescaleby = NODE_CLASS_MAPPINGS["ImageScaleBy"]()
        imagescaleby_27 = imagescaleby.upscale(
            upscale_method="lanczos",
            scale_by=1,
            image=get_value_at_index(loadimage_16, 0),
        )

        vaeencodetiled = NODE_CLASS_MAPPINGS["VAEEncodeTiled"]()
        vaeencodetiled_15 = vaeencodetiled.encode(
            tile_size=1024,
            overlap=64,
            temporal_size=64,
            temporal_overlap=8,
            pixels=get_value_at_index(imagescaleby_27, 0),
            vae=get_value_at_index(checkpointloadersimple_7, 2),
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_17 = cliptextencode.encode(
            text="masterpiece, best quality, highres",
            clip=get_value_at_index(loraloader_9, 1),
        )

        cliptextencode_18 = cliptextencode.encode(
            text="(worst quality, low quality, normal quality:1.5)",
            clip=get_value_at_index(loraloader_9, 1),
        )

        controlnetapplyadvanced = NODE_CLASS_MAPPINGS["ControlNetApplyAdvanced"]()
        freeu_v2 = NODE_CLASS_MAPPINGS["FreeU_V2"]()
        perturbedattentionguidance = NODE_CLASS_MAPPINGS["PerturbedAttentionGuidance"]()
        automatic_cfg = NODE_CLASS_MAPPINGS["Automatic CFG"]()
        tileddiffusion = NODE_CLASS_MAPPINGS["TiledDiffusion"]()
        samplercustom = NODE_CLASS_MAPPINGS["SamplerCustom"]()
        vaedecodetiled = NODE_CLASS_MAPPINGS["VAEDecodeTiled"]()

        for q in range(1):
            controlnetapplyadvanced_4 = controlnetapplyadvanced.apply_controlnet(
                strength=0.5,
                start_percent=0,
                end_percent=0.9,
                positive=get_value_at_index(cliptextencode_17, 0),
                negative=get_value_at_index(cliptextencode_18, 0),
                control_net=get_value_at_index(controlnetloader_3, 0),
                image=get_value_at_index(loadimage_16, 0),
            )

            freeu_v2_10 = freeu_v2.patch(
                b1=0.9,
                b2=1.08,
                s1=0.9500000000000001,
                s2=0.8,
                model=get_value_at_index(loraloader_9, 0),
            )

            perturbedattentionguidance_11 = perturbedattentionguidance.patch(
                scale=1, model=get_value_at_index(freeu_v2_10, 0)
            )

            automatic_cfg_12 = automatic_cfg.patch(
                hard_mode=True,
                boost=True,
                model=get_value_at_index(perturbedattentionguidance_11, 0),
            )

            tileddiffusion_13 = tileddiffusion.apply(
                method="MultiDiffusion",
                tile_width=1024,
                tile_height=1024,
                tile_overlap=128,
                tile_batch_size=4,
                model=get_value_at_index(automatic_cfg_12, 0),
            )

            samplercustom_20 = samplercustom.sample(
                add_noise=True,
                noise_seed=random.randint(1, 2**64),
                cfg=8,
                model=get_value_at_index(tileddiffusion_13, 0),
                positive=get_value_at_index(controlnetapplyadvanced_4, 0),
                negative=get_value_at_index(controlnetapplyadvanced_4, 1),
                sampler=get_value_at_index(ksamplerselect_2, 0),
                sigmas=get_value_at_index(alignyourstepsscheduler_1, 0),
                latent_image=get_value_at_index(vaeencodetiled_15, 0),
            )

            vaedecodetiled_5 = vaedecodetiled.decode(
                tile_size=1024,
                overlap=64,
                temporal_size=64,
                temporal_overlap=8,
                samples=get_value_at_index(samplercustom_20, 0),
                vae=get_value_at_index(checkpointloadersimple_7, 2),
            )
            res_image = tensor2pil(vaedecodetiled_5[0])
            img_input = tensor2pil(loadimage_16[0])
            debug_img = make_image_grid([img_input, res_image], cols=2, rows=1).convert('RGB')
            debug_img.save('/home/dell/study/test_comfy/img/v15_upscaled.png')

            


if __name__ == "__main__":
    main()
