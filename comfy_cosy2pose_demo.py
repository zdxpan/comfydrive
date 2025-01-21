import sys,os
import importlib.util
from typing import Sequence, Mapping, Any, Union
import asyncio
import logging
import torch 
import random
import argparse


comfyui_path = '/home/dell/study/comfyui/'
custom_node_path = comfyui_path + 'custom_nodes'

sys.path.append(comfyui_path)
os.chdir(comfyui_path)
from nodes import init_extra_nodes, init_builtin_extra_nodes, load_custom_node
import execution
import folder_paths
import server


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

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]

def load_extra_module(module_path: str, module_name: str = "extra_config") -> object:
    """使用 importlib 动态加载指定路径的模块。
    参数:
    - module_path: 模块文件的绝对路径。
    - module_name: 模块名称，默认为 'extra_config'。
    返回:
    - 加载的模块对象。
    """
    try:
        # 获取模块规范
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None:
            raise ImportError(f"Cannot find module at {module_path}")
        # 创建模块
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        # 执行模块
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logging.warning(f"Cannot import {module_path} module for extra_config: {e}")
        raise


try:
    from utils.extra_config import load_extra_path_config
except Exception as e:
    # 示例用法
    module_path = comfyui_path + "./utils/extra_config.py"
    module_ = load_extra_module(module_path, 'load_extra_path_config')
    print('load_extra_path_config' in sys.modules)
    extra_model_paths = find_path("extra_model_paths.yaml", comfyui_path)
    sys.modules['load_extra_path_config'].load_extra_path_config(extra_model_paths)

from nodes import NODE_CLASS_MAPPINGS
import_failed = init_builtin_extra_nodes()
# Creating a new event loop and setting it as the default loop
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
# Creating an instance of PromptServer with the loop
server_instance = server.PromptServer(loop)
execution.PromptQueue(server_instance)
custom_nodes = os.listdir(os.path.realpath(custom_node_path))
custom_nodes = ['ComfyUI_EchoMimic', 'CosyVoice-ComfyUI', 'ComfyUI-VideoHelperSuite']
base_node_names = set(NODE_CLASS_MAPPINGS.keys())

# some node need a PromptServer instance
for possible_module in custom_nodes:
    module_path = os.path.join(custom_node_path, possible_module) #comfyui/custom_nodes/ComfyUI_EchoMimic
    if os.path.isfile(module_path) and os.path.splitext(module_path)[1] != ".py": continue
    if module_path.endswith(".disabled"): continue
    success = load_custom_node(module_path, base_node_names, module_parent="custom_nodes")


# build gradio APP 
import gradio as gr
# from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
# from cosyvoice.utils.file_utils import load_wav, logging
inference_mode_list = ['预训练音色', '3s极速复刻', '跨语种复刻', '自然语言控制']
sft_spk_list = ['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']
stream_mode_list = [('否', False), ('是', True)]
FAMILY_NAMES = ['cosyvoice', 'echoMimic']
pose_path_list = NODE_CLASS_MAPPINGS["Echo_Sampler"]().INPUT_TYPES()['required']['pose_dir'][0]


cosyvoicenode = NODE_CLASS_MAPPINGS["CosyVoiceNode"]()

def generate_audio(tts_text, tts_mode, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, 
        instruct_text, seed, speed):
    if tts_mode != '预训练音色':
        gr.Info('Not support now')
    with torch.inference_mode():
        # waveform, sample_rate = torchaudio.load(audio_path)
        # audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        loadaudio = NODE_CLASS_MAPPINGS["LoadAudio"]()
        # loadaudio_13 = loadaudio.load(audio="audio_507729_temp.wav")   #  TODO  to temperature file and then read 
        loadaudio_13 = None
        if prompt_wav_upload:
            loadaudio_13 = loadaudio.load(audio=prompt_wav_upload)[0]
        # {'waveform': tensor([[[0., 0., 0.,  ..., 0., 0., 0.],
        #   [0., 0., 0.,  ..., 0., 0., 0.]]]),
        #  'sample_rate': 44100}
        cosyvoicenode = NODE_CLASS_MAPPINGS["CosyVoiceNode"]()
        saveaudio = NODE_CLASS_MAPPINGS["SaveAudio"]()

        for q in range(1):
            cosyvoicenode_15 = cosyvoicenode.generate(
                speed=speed,
                inference_mode=tts_mode, # "预训练音色",
                sft_dropdown=sft_dropdown,
                # seed=random.randint(1, 2**32),
                seed  = seed,
                tts_text=tts_text,
                prompt_text=prompt_text,
                prompt_wav=loadaudio_13,
            ) # 32767 = 2**16 / 2  
        res = (get_value_at_index(cosyvoicenode_15, 0)['sample_rate'], get_value_at_index(cosyvoicenode_15, 0)['waveform'].squeeze().numpy() * 32767)
        return gr.update(value=res)  # use all the path way


def generate_echo_video(image_input, audio_input, pose_input, width, height, length, steps, cfg, fps, context_frames, context_overlap, seed):
        if image_input is None or audio_input is None:
            return gr.update(value=None)
        echo_loadmodel = NODE_CLASS_MAPPINGS["Echo_LoadModel"]()
        echo_loadmodel_1 = echo_loadmodel.main_loader(
            vae="stabilityai/sd-vae-ft-mse",denoising=True,infer_mode="pose_normal_dwpose",
            draw_mouse=False,motion_sync=False,lowvram=False,version="V2",
        )

        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_3 = loadimage.load_image(image="11f297aed83799aaa6e54aede4570292.jpg")

        loadaudio = NODE_CLASS_MAPPINGS["LoadAudio"]()
        loadaudio_4 = loadaudio.load(audio="ultraman.wav")

        vhs_loadvideo = NODE_CLASS_MAPPINGS["VHS_LoadVideo"]()
        vhs_loadvideo_11 = None
        # vhs_loadvideo_11 = vhs_loadvideo.load_video(
        #     video="賽琳娜戈梅茲- 像情歌一樣愛你Selena Gomez-Love You Like A Love Song_A1.art_.mp4",
        #     force_rate=0,force_size="Disabled",
        #     custom_width=512,custom_height=512,frame_load_cap=60,
        #     skip_first_frames=0,select_every_nth=1,unique_id=6076199186146585561,
        # )

        vhs_videoinfo = NODE_CLASS_MAPPINGS["VHS_VideoInfo"]()
        echo_sampler = NODE_CLASS_MAPPINGS["Echo_Sampler"]()
        vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()

        for q in range(1):
            vhs_videoinfo_12 = vhs_videoinfo.get_video_info(    # (30.0, 450, 15.0, 832, 1152, 30.0, 60, 2.0, 832, 1152)
                video_info=get_value_at_index(vhs_loadvideo_11, 3)
            )
            echo_sampler_13 = echo_sampler.em_main(
                pose_dir=pose_input,    #  pose_input 'assets/halfbody_demo/pose/fight'
                seed=random.randint(1, 2**32),
                cfg=2.5, steps=steps, 
                fps=get_value_at_index(vhs_videoinfo_12, 0),
                sample_rate=16000,
                facemask_ratio=0.1,
                facecrop_ratio=0.8,
                context_frames=context_frames, context_overlap=context_overlap, length=length,  # total frames
                width=width, height=height,  save_video=False,
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
            # vhs_videocombine_17['result'][0][1]
            # ['/home/dell/study/comfyui/output/AnimateDiff_00005.png', 
            # '/home/dell/study/comfyui/output/AnimateDiff_00005.mp4', 
            # '/home/dell/study/comfyui/output/AnimateDiff_00005-audio.mp4']
        return gr.update(value=vhs_videocombine_17['result'][0][1][2])

class EchoUI:
    def setup_ui(self):
        with gr.Tab(label="Echo Avatar"):
            with gr.Row():
                with gr.Column(scale=6):
                    video_output = gr.Video(height=512, visible=True)
                    with gr.Row(elem_classes='scale-1'):
                        meta = gr.JSON(label='Meta')
                        args = gr.JSON(label='Parameters')
                
                with gr.Column(scale=4, variant='panel'):
                    with gr.Group():
                        image_input = gr.Image(label="image", type="filepath", min_width=512)
                        with gr.Row():
                            audio_input = gr.Audio(label="audio", sources=["upload", "microphone"], type="filepath")
                        with gr.Row():
                            pose_input = gr.Dropdown(label="姿态输入（目录地址）", value=pose_path_list[0], choices=pose_path_list)
                    with gr.Row():
                        generate_button = gr.Button("🎬 Echo_Run Video")
                    with gr.Row():
                        width = gr.Number(label="width", value=768)
                        height = gr.Number(label="height", value=768)
                        length = gr.Number(label="video length 120", value=120)

                    with gr.Row():
                        steps = gr.Number(label="steps", value=20)
                        cfg = gr.Number(label="cfg 2.5", value=2.5, step=0.1)
                        seed = gr.Number(label="种子(-1为随机)", value=-1)
                    with gr.Row():
                        fps = gr.Number(label="fps", value=24)
                        context_frames = gr.Number(label="context lenghth", value=12)
                        context_overlap = gr.Number(label="上下文重叠（默认3）", value=3)
                        # sample_rate = gr.Number(label="采样率（默认16000）", value=16000)
            # gr.Examples(
            #     examples=[
            #         ["EMTD_dataset/ref_imgs_by_FLUX/man/0003.png", "assets/halfbody_demo/audio/chinese/fighting.wav"],
            #         ["EMTD_dataset/ref_imgs_by_FLUX/woman/0033.png", "assets/halfbody_demo/audio/chinese/good.wav"],
            #         ["EMTD_dataset/ref_imgs_by_FLUX/man/0010.png", "assets/halfbody_demo/audio/chinese/news.wav"],
            #         ["EMTD_dataset/ref_imgs_by_FLUX/man/1168.png", "assets/halfbody_demo/audio/chinese/no_smoking.wav"],
            #         ["EMTD_dataset/ref_imgs_by_FLUX/woman/0057.png", "assets/halfbody_demo/audio/chinese/ultraman.wav"],
            #         ["EMTD_dataset/ref_imgs_by_FLUX/man/0001.png", "assets/halfbody_demo/audio/chinese/echomimicv2_man.wav"],
            #         ["EMTD_dataset/ref_imgs_by_FLUX/woman/0077.png", "assets/halfbody_demo/audio/chinese/echomimicv2_woman.wav"],
            #     ],
            #     inputs=[image_input, audio_input],  
            #     label="预设人物及音频",
            # )
        generate_button.click(
            generate_echo_video,
            inputs=[image_input, audio_input, pose_input, width, height, length, steps, cfg, fps, context_frames, context_overlap, seed],
            outputs=[video_output],
        )

class WebUI:   # 总的层级
    css = ' '.join([
        '.scale-1 {flex-grow:1;}',
        '.accordion-panel {background:var(--panel-background-fill);border:none !important;}',
    ])
    def __init__(self, args: argparse.Namespace):
        # super().__init__()
        self.demo = gr.Blocks(theme=gr.themes.Base(), css=self.css, analytics_enabled=False).queue()
        self.echo_tab = EchoUI()
        # self.trainer_tab = TrainerTab(args)
        # self.exporter_tab = ExporterTab(args)
        self.setup_ui()
    def setup_ui(self):
        with self.demo:   # 按照平铺展开  , 每个子模块的UI 按照 with gr.Tab(label="Visualize"):  展开
            self.basic_ui()
            self.echo_tab.setup_ui()
            # self.trainer_tab.setup_ui()
            # self.exporter_tab.setup_ui()
    def launch(self, **kwargs):
        self.demo.launch(**kwargs)
    
    def basic_ui(self):
        with gr.Row():
            gr.HTML('<h3 style="margin:0"> 🎡 PLAYGROUND</h3>')
        with gr.Tab(label="CosyTTs"):
            with gr.Row():
                with gr.Column(scale=2, variant='panel'):
                    gallery = gr.Gallery(label='Gallery', height=400, show_label=False, format='jpeg')
                    audio_output = gr.Audio(label="合成音频", autoplay=True, streaming=True)
                    video = gr.Video(height=512, visible=False)
                    with gr.Row(elem_classes='scale-1'):
                        meta = gr.JSON(label='Meta')
                        args = gr.JSON(label='Parameters')

                with gr.Column(scale=1):
                    # family = gr.Dropdown(FAMILY_NAMES, label='Family', value=FAMILY_NAMES[1], container=False)
                    tts_text = gr.Textbox(label="Input text", lines=1, value="生成式语音大模型，提供舒适自然的语音合成能力。")
                    instruct_text = gr.Textbox(label="instruct text", lines=1, value='')
                    with gr.Row():
                        tts_mode = gr.Dropdown(inference_mode_list, label='TTS-mode', value=inference_mode_list[0], container=False)
                        # instruction_text = gr.Text(label="操作步骤", value=instruct_dict[inference_mode_list[0]], scale=0.5)
                        sft_dropdown = gr.Dropdown(choices=sft_spk_list, label='选择预训练音色', value=sft_spk_list[0], scale=0.25, container=False)
                    with gr.Row():
                        # stream = gr.Checkbox(value=False, label='是否流式推理')
                        speed = gr.Number(value=1, label="速度调节(仅支持非流式推理)", minimum=0.5, maximum=2.0, step=0.1)
                        seed = gr.Number(value=0, label="随机推理种子")
                    with gr.Row():
                        prompt_text = gr.Textbox(label="prompt文本", lines=1, placeholder="请输入prompt文本，需与prompt音频内容一致，暂时不支持自动识别...", value='')
                    with gr.Row():
                        prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='选择prompt音频文件，注意采样率不低于16khz', scale=1)
                    with gr.Accordion(label="reord", open=False):
                        prompt_wav_record = gr.Audio(sources=['microphone'], type='filepath', label='录制prompt音频文件', scale=1)

                    # with gr.Tab('Prompt'):
                    #     prompt = gr.Textbox(lines=3, container=False, show_label=False)
                    #     instruct_text = 
                    with gr.Row():
                        generate_button = gr.Button("生成音频")
        
        generate_button.click(generate_audio,
            inputs=[tts_text, tts_mode, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                    seed, speed],
            outputs=[audio_output])

demo = WebUI(args=None).launch(server_name='0.0.0.0', server_port=5002)

if __name__ == "__main__1":

    with torch.inference_mode():
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
                seed=random.randint(1, 2**32),
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



