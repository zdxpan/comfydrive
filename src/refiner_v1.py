# must run at custom_nodes after custom nodes loaded!
from util import (
    find_path, add_comfyui_directory_to_sys_path, add_extra_model_paths,
    tensor2pil, pil2tensor, pilmask2tensor, make_image_grid, draw_text, zdxApplySageAtt
)

from nodes import NODE_CLASS_MAPPINGS


class Refinerv1():
    def __init__(self):
        self.name = self.__class__.__name__
        with torch.inference_mode():
            self.clipvisionloader = NODE_CLASS_MAPPINGS["CLIPVisionLoader"]()
            self.imageresizekj = NODE_CLASS_MAPPINGS["ImageResizeKJ"]()
            self.imagescaleby = NODE_CLASS_MAPPINGS["ImageScaleBy"]()

            self.imageupscalewithmodel = NODE_CLASS_MAPPINGS["ImageUpscaleWithModel"]()
            
            self.loraloader = NODE_CLASS_MAPPINGS["LoraLoader"]()
            self.vaeencodetiled = NODE_CLASS_MAPPINGS["VAEEncodeTiled"]()
            self.controlnetloader = NODE_CLASS_MAPPINGS["ControlNetLoader"]()
            self.controlnetloader_697 = self.controlnetloader.load_controlnet(
                control_net_name="control_v11f1e_sd15_tile.pth"
            )
            self.freeu_v2 = NODE_CLASS_MAPPINGS["FreeU_V2"]()
            self.perturbedattentionguidance = NODE_CLASS_MAPPINGS["PerturbedAttentionGuidance"]()
            self.automatic_cfg = NODE_CLASS_MAPPINGS["Automatic CFG"]()
            self.tileddiffusion = NODE_CLASS_MAPPINGS["TiledDiffusion"]()
            self.controlnetapplyadvanced = NODE_CLASS_MAPPINGS["ControlNetApplyAdvanced"]()
            self.samplercustom = NODE_CLASS_MAPPINGS["SamplerCustom"]()
            self.vaedecodetiled = NODE_CLASS_MAPPINGS["VAEDecodeTiled"]()
            self.imagescale = NODE_CLASS_MAPPINGS["ImageScale"]()

            self.checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
            self.checkpointloadersimple_685 = self.checkpointloadersimple.load_checkpoint(
                ckpt_name="juggernaut_reborn.safetensors"
            )
            loraloader_687 = self.loraloader.load_lora(
                lora_name="v15/more_details.safetensors",
                strength_model=0.2,
                strength_clip=0.2,
                model=get_value_at_index(self.checkpointloadersimple_685, 0),
                clip=get_value_at_index(self.checkpointloadersimple_685, 1),
            )

            self.loraloader_686 = self.loraloader.load_lora(
                lora_name="v15/SDXLrender_v2.0.safetensors",
                strength_model=0.1,
                strength_clip=0.2,
                model=get_value_at_index(loraloader_687, 0),
                clip=get_value_at_index(loraloader_687, 1),
            )
            self.freeu_v2_691 = self.freeu_v2.patch(
                b1=0.9,
                b2=1.08,
                s1=0.9500000000000001,
                s2=0.8,
                model=get_value_at_index(self.loraloader_686, 0),
            )
            self.perturbedattentionguidance_675 = self.perturbedattentionguidance.patch(
                scale=1, model=get_value_at_index(self.freeu_v2_691, 0)
            )
            self.automatic_cfg_688 = self.automatic_cfg.patch(
                hard_mode=True,
                boost=True,
                model=get_value_at_index(self.perturbedattentionguidance_675, 0),
            )
            self.tileddiffusion_698 = self.tileddiffusion.apply(
                method="MultiDiffusion",
                tile_width=1024,
                tile_height=1024,
                tile_overlap=128,
                tile_batch_size=4,
                model=get_value_at_index(self.automatic_cfg_688, 0),
            )
            self.cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
            self.cliptextencode_676 = self.cliptextencode.encode(
                text="(worst quality, low quality, normal quality:1.5)",
                clip=get_value_at_index(self.loraloader_686, 1),
            )
            self.cliptextencode_684 = self.cliptextencode.encode(
                text="masterpiece, best quality, highres",
                clip=get_value_at_index(self.loraloader_686, 1),
            )

            self.alignyourstepsscheduler = NODE_CLASS_MAPPINGS["AlignYourStepsScheduler"]()

            ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
            self.ksamplerselect_678 = ksamplerselect.get_sampler(sampler_name="dpmpp_3m_sde_gpu")

            self.upscalemodelloader = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
            self.upscalemodelloader_690 = self.upscalemodelloader.load_model(
                model_name="RealESRGAN_x2plus.pth"
            )
            self.imagescaleby = NODE_CLASS_MAPPINGS["ImageScaleBy"]()

    def forward(self, input_image):
        with torch.inference_mode():
            # upscale use esrgan x2
            imageupscalewithmodel_695 = self.imageupscalewithmodel.upscale(
                upscale_model=get_value_at_index(self.upscalemodelloader_690, 0),
                image=get_value_at_index(input_image, 0),
            )
            
            imagescaleby_694 = self.imagescaleby.upscale(
                upscale_method="lanczos",
                scale_by=0.6000000000000001,
                image=get_value_at_index(imageupscalewithmodel_695, 0),
            )
            # encode as latent
            vaeencodetiled_693 = self.vaeencodetiled.encode(
                tile_size=1024,
                overlap=64,
                temporal_size=64,
                temporal_overlap=8,
                pixels=get_value_at_index(imagescaleby_694, 0),
                vae=get_value_at_index(self.checkpointloadersimple_685, 2),
            )

            alignyourstepsscheduler = NODE_CLASS_MAPPINGS["AlignYourStepsScheduler"]()
            alignyourstepsscheduler_677 = alignyourstepsscheduler.get_sigmas(
                model_type="SD1", steps=30, denoise=0.4
            )
            controlnetapplyadvanced_699 = self.controlnetapplyadvanced.apply_controlnet(
                strength=1,
                start_percent=0.1,
                end_percent=1,
                positive=get_value_at_index(self.cliptextencode_684, 0),
                negative=get_value_at_index(self.cliptextencode_676, 0),
                control_net=get_value_at_index(self.controlnetloader_697, 0),
                image=get_value_at_index(input_image, 0),
            )

            samplercustom_689 = self.samplercustom.sample(
                add_noise=True,
                noise_seed=random.randint(1, 2**64),
                cfg=8,
                model=get_value_at_index(self.tileddiffusion_698, 0),
                positive=get_value_at_index(controlnetapplyadvanced_699, 0),
                negative=get_value_at_index(controlnetapplyadvanced_699, 1),
                sampler=get_value_at_index(self.ksamplerselect_678, 0),
                sigmas=get_value_at_index(alignyourstepsscheduler_677, 0),
                latent_image=get_value_at_index(vaeencodetiled_693, 0),
            )

            vaedecodetiled_679 = self.vaedecodetiled.decode(
                tile_size=1024,
                overlap=64,
                temporal_size=64,
                temporal_overlap=8,
                samples=get_value_at_index(samplercustom_689, 0),
                vae=get_value_at_index(self.checkpointloadersimple_685, 2),
            )
        # use original struct return
        return vaedecodetiled_679

