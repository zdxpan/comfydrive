import json
import logging
import sys
import torch
from typing import Dict, Any, Optional, List, Set

from nodes import init_extra_nodes, NODE_CLASS_MAPPINGS
from comfy_execution.graph import DynamicPrompt, ExecutionList
from comfy_execution.caching import HierarchicalCache, CacheKeySetInputSignature, CacheKeySetID
from execution import validate_prompt, get_input_data, get_output_data, _map_node_over_list, ExecutionBlocker
from convert_workflow import convert_workflow_to_api

def setup_comfyui_env():
    """Setup ComfyUI environment and initialize nodes"""

    # Setup logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True,
        handlers=[
            logging.FileHandler('comfyui_execution.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Initialize nodes including custom nodes
    init_extra_nodes()

class IsChangedCache:
    """Cache for tracking node changes"""
    def __init__(self, dynprompt, outputs_cache):
        self.dynprompt = dynprompt
        self.outputs_cache = outputs_cache
        self.is_changed = {}

    def get(self, node_id: str) -> Any:
        """Get change status for a node"""
        if node_id in self.is_changed:
            return self.is_changed[node_id]

        node = self.dynprompt.get_node(node_id)
        class_type = node["class_type"]
        class_def = NODE_CLASS_MAPPINGS[class_type]

        if not hasattr(class_def, "IS_CHANGED"):
            self.is_changed[node_id] = False
            return False

        if "is_changed" in node:
            self.is_changed[node_id] = node["is_changed"]
            return node["is_changed"]

        input_data_all, _ = get_input_data(node["inputs"], class_def, node_id, None)
        try:
            is_changed = _map_node_over_list(class_def, input_data_all, "IS_CHANGED")
            node["is_changed"] = [None if isinstance(x, ExecutionBlocker) else x for x in is_changed]
        except Exception as e:
            logging.warning(f"WARNING: {e}")
            node["is_changed"] = float("NaN")
        finally:
            self.is_changed[node_id] = node["is_changed"]
        return self.is_changed[node_id]


class ExecutionCache:
    """Cache management for workflow execution"""
    def __init__(self):
        self.outputs = HierarchicalCache(CacheKeySetInputSignature)
        self.ui = HierarchicalCache(CacheKeySetInputSignature)
        self.objects = HierarchicalCache(CacheKeySetID)
        self._caches = [self.outputs, self.ui, self.objects]

    def initialize(self, prompt: Dict, dynamic_prompt: DynamicPrompt) -> None:
        """Initialize all caches with the prompt"""
        is_changed_cache = IsChangedCache(dynamic_prompt, self.outputs)
        for cache in self._caches:
            cache.set_prompt(dynamic_prompt, set(prompt.keys()), is_changed_cache)
            cache.clean_unused()

    def get_output(self, node_id: str) -> Optional[Any]:
        """Get cached output for a node"""
        return self.outputs.get(node_id)

    def set_output(self, node_id: str, output: Any, ui_data: Optional[Dict] = None) -> None:
        """Set output and UI data for a node"""
        self.outputs.set(node_id, output)
        if ui_data:
            self.ui.set(node_id, {
                "meta": {"node_id": node_id},
                "output": ui_data
            })

    def get_object(self, node_id: str) -> Optional[Any]:
        """Get cached object instance"""
        return self.objects.get(node_id)

    def set_object(self, node_id: str, obj: Any) -> None:
        """Cache object instance"""
        self.objects.set(node_id, obj)

class WorkflowExecutor:
    def __init__(self):
        self.cache = ExecutionCache()
        self.executed: Set[str] = set()
        self.execution_order: List[str] = []

    def _execute_node(self, node_id: str, dynamic_prompt: DynamicPrompt, execution_list: ExecutionList) -> Optional[Any]:
        """Execute a single node in the workflow"""
        if node_id in self.executed:
            return self.cache.get_output(node_id)

        try:
            node = dynamic_prompt.get_node(node_id)
            class_type = node['class_type']
            class_def = NODE_CLASS_MAPPINGS[class_type]

            # Get or create node instance
            node_instance = self.cache.get_object(node_id)
            if node_instance is None:
                node_instance = class_def()
                self.cache.set_object(node_id, node_instance)

            logging.info(f"Executing node {node_id} ({class_type})")

            # Get input data
            input_data_all, missing = get_input_data(
                node['inputs'],
                class_def,
                node_id,
                self.cache.outputs,
                dynprompt=dynamic_prompt,
                extra_data={}
            )

            # Handle lazy inputs
            if hasattr(node_instance, "check_lazy_status"):
                required_inputs = _map_node_over_list(node_instance, input_data_all, "check_lazy_status", allow_interrupt=True)
                required_inputs = set(sum([r for r in required_inputs if isinstance(r,list)], []))
                required_inputs = [x for x in required_inputs if isinstance(x,str) and (
                    x not in input_data_all or x in missing
                )]
                if required_inputs:
                    for input_name in required_inputs:
                        execution_list.make_input_strong_link(node_id, input_name)
                    return None

            # Execute node function
            output_data, ui_data, has_subgraph = get_output_data(
                node_instance,
                input_data_all,
                execution_block_cb=None
            )

            if has_subgraph:
                raise ValueError("Subgraph execution not supported in standalone mode")

            # Cache results
            self.cache.set_output(node_id, output_data, ui_data)

            logging.info(f"Executed node {node_id} ({class_type})")

            return output_data

        except Exception as ex:
            logging.error(f"Error executing node {node_id} ({class_type}): {str(ex)}")
            raise

    def execute(self, workflow_json: str | dict) -> Dict[str, Any]:
        """Execute a complete workflow from JSON"""
        prompt = json.loads(workflow_json) if isinstance(workflow_json, str) else workflow_json

        # Validate workflow
        valid, error, outputs, _ = validate_prompt(prompt)
        if not valid:
            raise ValueError(f"Invalid workflow: {error['message']}\n{error['details']}")

        if not outputs:
            raise ValueError("No output nodes found in workflow")

        # Initialize execution
        dynamic_prompt = DynamicPrompt(prompt)
        self.cache.initialize(prompt, dynamic_prompt)
        execution_list = ExecutionList(dynamic_prompt, self.cache.outputs)

        # Add output nodes to execution list
        for node_id in outputs:
            execution_list.add_node(node_id)

        # Execute workflow
        with torch.inference_mode():
            while not execution_list.is_empty():
                node_id, error, _ = execution_list.stage_node_execution()
                if error:
                    raise RuntimeError(f"Execution error: {error['exception_message']}")

                output = self._execute_node(node_id, dynamic_prompt, execution_list)
                if output is None:
                    execution_list.unstage_node_execution()
                else:
                    if node_id not in self.executed:
                        self.execution_order.append(node_id)
                        self.executed.add(node_id)
                    execution_list.complete_node_execution()

        return {
            'outputs': self.cache.outputs.recursive_debug_dump(),
            'executed_nodes': self.execution_order
        }

def execute_workflow(workflow_json: str | dict) -> Dict[str, Any]:
    """Helper function to execute a workflow"""
    executor = WorkflowExecutor()
    return executor.execute(workflow_json)

if __name__ == "__main__":
    setup_comfyui_env()
    # Change as required     
    workflow_path = "/home/ubuntu/ComfyUI/user/default/workflows/sdxl_workflow_api.json"
    logging.info(f"Loading workflow from: {workflow_path}")
    with open(workflow_path) as json_data:
        workflow_json = json.load(json_data)
    results = execute_workflow(workflow_json)
    logging.info("Workflow executed successfully")
    logging.info("Executed nodes: %s", results['executed_nodes'])