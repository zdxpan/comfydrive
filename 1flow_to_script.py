import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch

__path = '/home/dell/study/comfyui'
sys.path.append(__path)

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
# add extra_path.yaml 
extra_model_paths = find_path("extra_model_paths.yaml", __path)
custom_nodes = ['ComfyUI_EchoMimic', 'CosyVoice-ComfyUI', 'ComfyUI-VideoHelperSuite']
custom_nodes = []
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
    from nodes import init_extra_nodes, init_builtin_extra_nodes, load_custom_node, init_external_custom_nodes
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
    if custom_nodes is None or len(custom_nodes) == 0:
        init_external_custom_nodes()  # it will load all node, not recommend
        return
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


# way |
import importlib
from nodes import init_extra_nodes, init_builtin_extra_nodes, load_custom_node, init_external_custom_nodes
module_path = '/home/dell/study/comfyui/custom_nodes/ComfyUI-to-Python-Extension/'
import_custom_nodes([module_path])
from comfyui_to_python import ComfyUItoPython

# way ||
import_custom_nodes(custom_nodes)
from comfyui_to_python import ComfyUItoPython


ComfyUItoPython(
    input_file='/home/dell/study/comfyui/1workflows/base_cosy_workflow_api.json',
    output_file='/home/dell/study/comfyui/1workflows/base_cosy_workflow_api',
    queue_size=1,
    # node_class_mappings=NODE_CLASS_MAPPINGS,
    needs_init_custom_nodes=False,
    )
