from comfy_script.runtime import *
load()
# load('http://127.0.0.1:8188/')
from comfy_script.runtime.nodes import *

with Workflow():
    model, clip, vae = CheckpointLoaderSimple('v1-5-pruned-emaonly.ckpt')
    conditioning = CLIPTextEncode('beautiful scenery nature glass bottle landscape, , purple galaxy bottle,', clip)
    conditioning2 = CLIPTextEncode('text, watermark', clip)
    latent = EmptyLatentImage(512, 512, 1)
    latent = KSampler(model, 156680208700286, 20, 8, 'euler', 'normal', conditioning, conditioning2, latent, 1)
    image = VAEDecode(latent, vae)
    SaveImage(image, 'ComfyUI')




import comfy_script.runtime as runtime
runtime.start_comfyui(no_server=True, autonomy=True)

# unload models
import comfy.model_management

model, clip, vae = CheckpointLoaderSimple(Checkpoints.v1_5_pruned_emaonly)
conditioning = CLIPTextEncode('text, watermark', clip)

print(comfy.model_management.current_loaded_models)
# [<comfy.model_management.LoadedModel object at 0x0000014C2287EF80>, <comfy.model_management.LoadedModel object at 0x0000014C2287EB00>]

comfy.model_management.unload_model_clones(model)
comfy.model_management.unload_model_clones(clip.patcher)
comfy.model_management.unload_model_clones(vae.patcher)
print(comfy.model_management.current_loaded_models)
