import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch

comfyui_path = '/home/dell/study/comfyui'
sys.path.append(comfyui_path)

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


# 递归的find_path of comfyui 而不是显示的指定 路径
def find_path(name: str, path: str = None) -> str:
    path = os.getcwd() if path is None else path
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        return path_name
    # go to Parent dir
    parent_directory = os.path.dirname(path)
    if parent_directory == path:
        return None  # in root dir
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
# add extra_path.yaml 
extra_model_paths = find_path("extra_model_paths.yaml", comfyui_path)
custom_nodes = ['ComfyUI_EchoMimic', 'CosyVoice-ComfyUI', 'ComfyUI-VideoHelperSuite']
from nodes import NODE_CLASS_MAPPINGS
from utils.extra_config import load_extra_path_config
if extra_model_paths is not None:
    load_extra_path_config(extra_model_paths)  # load checkpoints vae clip  loras xlabels


def import_custom_nodes(custom_nodes) -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes, init_builtin_extra_nodes, load_custom_node
    import folder_paths
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    import_failed = init_builtin_extra_nodes()
    if import_failed:
        print("Failed to import some extra_ nodes.")
    # init_external_custom_nodes()  # it will load all node, not recommend
    base_node_names = set(NODE_CLASS_MAPPINGS.keys())
    node_paths = folder_paths.get_folder_paths("custom_nodes") # comfyui/custom_nodes
    custom_node_path = node_paths[0]
    # possible_modules = os.listdir(os.path.realpath(custom_node_path))
    for possible_module in custom_nodes:
        module_path = os.path.join(custom_node_path, possible_module) #comfyui/custom_nodes/ComfyUI_EchoMimic
        if os.path.isfile(module_path) and os.path.splitext(module_path)[1] != ".py": continue
        if module_path.endswith(".disabled"): continue
        success = load_custom_node(module_path, base_node_names, module_parent="custom_nodes")

    # init_extra_nodes()  # load all node and custom_nodes



def main():
    import_custom_nodes(custom_nodes)
    with torch.inference_mode():
        textnode = NODE_CLASS_MAPPINGS["TextNode"]()
        textnode_2 = textnode.encode(
            text="收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
        )

        textnode_12 = textnode.encode(text="希望你以后能够做的比我还好呦。")

        loadaudio = NODE_CLASS_MAPPINGS["LoadAudio"]()
        loadaudio_13 = loadaudio.load(audio="audio_507729_temp.wav")

        cosyvoicenode = NODE_CLASS_MAPPINGS["CosyVoiceNode"]()
        saveaudio = NODE_CLASS_MAPPINGS["SaveAudio"]()

        for q in range(1):
            cosyvoicenode_15 = cosyvoicenode.generate(
                speed=1.5,
                inference_mode="预训练音色",
                sft_dropdown="中文女",
                seed=random.randint(1, 2**64),
                tts_text=get_value_at_index(textnode_2, 0),
                prompt_text=get_value_at_index(textnode_12, 0),
                prompt_wav=get_value_at_index(loadaudio_13, 0),
            )

            saveaudio_14 = saveaudio.save_audio(
                filename_prefix="audio/ComfyUI",
                audio=get_value_at_index(cosyvoicenode_15, 0),
            )
        echo_loadmodel = NODE_CLASS_MAPPINGS["Echo_LoadModel"]()
        echo_loadmodel_1 = echo_loadmodel.main_loader(
            vae="stabilityai/sd-vae-ft-mse",
            denoising=True,
            infer_mode="pose_normal_dwpose",
            draw_mouse=False,
            motion_sync=False,
            lowvram=False,
            version="V2",
        )

        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_3 = loadimage.load_image(image="11f297aed83799aaa6e54aede4570292.jpg")

        loadaudio = NODE_CLASS_MAPPINGS["LoadAudio"]()
        loadaudio_4 = loadaudio.load(audio="ultraman.wav")

        vhs_loadvideo = NODE_CLASS_MAPPINGS["VHS_LoadVideo"]()
        vhs_loadvideo_11 = vhs_loadvideo.load_video(
            video="賽琳娜戈梅茲- 像情歌一樣愛你Selena Gomez-Love You Like A Love Song_A1.art_.mp4",
            force_rate=0,
            force_size="Disabled",
            custom_width=512,
            custom_height=512,
            frame_load_cap=60,
            skip_first_frames=0,
            select_every_nth=1,
            unique_id=6076199186146585561,
        )

        vhs_videoinfo = NODE_CLASS_MAPPINGS["VHS_VideoInfo"]()
        echo_sampler = NODE_CLASS_MAPPINGS["Echo_Sampler"]()
        vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()

        for q in range(1):
            vhs_videoinfo_12 = vhs_videoinfo.get_video_info(
                video_info=get_value_at_index(vhs_loadvideo_11, 3)
            )

            echo_sampler_13 = echo_sampler.em_main(
                pose_dir="pose_01",
                seed=random.randint(1, 2**64),
                cfg=2.5,
                steps=30,
                fps=get_value_at_index(vhs_videoinfo_12, 0),
                sample_rate=16000,
                facemask_ratio=0.1,
                facecrop_ratio=0.8,
                context_frames=12,
                context_overlap=3,
                length=50,
                width=768,
                height=768,
                save_video=False,
                image=get_value_at_index(loadimage_3, 0),
                audio=get_value_at_index(loadaudio_4, 0),
                model=get_value_at_index(echo_loadmodel_1, 0),
                face_detector=get_value_at_index(echo_loadmodel_1, 1),
                visualizer=get_value_at_index(echo_loadmodel_1, 2),
                video_images=get_value_at_index(vhs_loadvideo_11, 0),
            )

            vhs_videocombine_17 = vhs_videocombine.combine_video(
                frame_rate=get_value_at_index(echo_sampler_13, 2),
                loop_count=0,
                filename_prefix="AnimateDiff",
                format="video/h264-mp4",
                pix_fmt="yuv420p",
                crf=19,
                save_metadata=True,
                trim_to_audio=False,
                pingpong=False,
                save_output=True,
                images=get_value_at_index(echo_sampler_13, 0),
                audio=get_value_at_index(echo_sampler_13, 1),
                unique_id=15025088326710150899,
            )
            # vhs_videocombine_17.keys()  'ui', 'result'
            vhs_videocombine_17['result'][0][1]


if __name__ == "__main__":
    main()
