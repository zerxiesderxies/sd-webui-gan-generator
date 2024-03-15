from __future__ import annotations
from typing import Union
import sys
import os
import ast
from PIL import Image
from pathlib import Path

import gradio as gr
from modules import script_callbacks, shared, ui, ui_components
from modules.ui_components import ToolButton

from lib_gan_extension import global_state, file_utils, str_utils, metadata, GanGenerator, logger
ui.swap_symbol = "\U00002194"  # ↔️
ui.lucky_symbol = "\U0001F340"  # 🍀
ui.folder_symbol = "\U0001F4C1"  # 📁

model = GanGenerator()

DESCRIPTION = '''# StyleGAN Image Generator

Use this tool to generate random images with a pretrained StyleGAN3 network of your choice. 
Download model pickle files and place them in sd-webui-gan-generator/models folder. 
Supports generation with the cpu or gpu0. See available pretrained networks via [https://github.com/NVlabs/stylegan3](https://github.com/NVlabs/stylegan3).
Recommend using stylegan3-r-ffhq or stylegan2-celebahq
'''

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False, css='style.css') as ui_component:
        gr.Markdown(DESCRIPTION)
        with gr.Row():
            modelDrop = gr.Dropdown(choices = update_model_list(), value=default_model, label="Model Selection", info="Place into models directory", elem_id="models")
            modelDrop.input(fn=touch_model_file, inputs=[modelDrop], outputs=[])

            model_refreshButton = ToolButton(value=ui.refresh_symbol, tooltip="Refresh")
            model_refreshButton.click(fn=lambda: gr.Dropdown.update(choices=update_model_list()),outputs=modelDrop)

            with gr.Group():
                with gr.Column():
                    gr.Markdown(label='Output Folder', value="Output folder", elem_id="output-folder")
                    folderButton = ToolButton(ui.folder_symbol, visible=not shared.cmd_opts.hide_ui_dir_config, tooltip="Open image output directory", elem_id="open-folder")
                    folderButton.click(
                        fn=lambda images, index: file_utils.open_folder(model.outputRoot),
                        inputs=[],
                        outputs=[],
                    )

        with gr.Tabs():
            with gr.TabItem('Simple Image Gen', elem_id="simple-tab"):
                with gr.Row():
                    with gr.Column():
                        psiSlider = gr.Slider(-1,1,
                                        step=0.05,
                                        value=0.7,
                                        label='Truncation (psi)')
                        with gr.Row():
                            seedNum = gr.Number(label='Seed', value=lambda: -1, min_width=150, precision=0)

                            seed_randButton = ToolButton(ui.random_symbol, tooltip="Set seed to -1, which will cause a new random number to be used every time")
                            seed_randButton.click(fn=lambda: seedNum.update(value=-1), show_progress=False, inputs=[], outputs=[seedNum])

                            seed_recycleButton = ToolButton(ui.reuse_symbol, tooltip="Reuse seed from last generation")

                        simple_runButton = gr.Button('Generate Simple Image', variant="primary", id="simple_generate")

                    with gr.Column():
                        resultImg = gr.Image(label='Result', sources=['upload','clipboard'], interactive=True, type="filepath", elem_classes="gan-output")
                        resultImg.upload(
                            fn=get_params_from_image,
                            inputs=[resultImg],
                            outputs=[seedNum,psiSlider],
                            show_progress=False
                        )
                        seedTxt = gr.Markdown(label='Output Seed')
                        with gr.Row():
                            seed1_to_mixButton = gr.Button('Send to Seed Mixer › Left')
                            seed2_to_mixButton = gr.Button('Send to Seed Mixer › Right')

            with gr.TabItem('Seed Mixer', elem_id="mix-tab"):
                with gr.Row():
                    mix_seed1_Num = gr.Number(label='Seed 1', value=lambda: -1, min_width=150, precision=0)

                    mix_seed1_luckyButton = ToolButton(ui.lucky_symbol, tooltip="Roll generate a new seed")
                    mix_seed1_luckyButton.click(fn=lambda: mix_seed1_Num.update(value=model.newSeed()), show_progress=False, inputs=[], outputs=[mix_seed1_Num])

                    mix_seed1_randButton = ToolButton(ui.random_symbol, tooltip="Set seed to -1, which will cause a new random number to be used every time")
                    mix_seed1_randButton.click(fn=lambda: mix_seed1_Num.update(value=-1), show_progress=False, inputs=[], outputs=[mix_seed1_Num])

                    mix_seed1_recycleButton = ToolButton(ui.reuse_symbol, tooltip="Reuse seed from last generation")

                    mix_seed2_Num = gr.Number(label='Seed 2', value=lambda: -1, min_width=150, precision=0)

                    mix_seed2_luckyButton = ToolButton(ui.lucky_symbol, tooltip="Roll generate a new seed")
                    mix_seed2_luckyButton.click(fn=lambda: mix_seed2_Num.update(value=model.newSeed()), show_progress=False, inputs=[], outputs=[mix_seed2_Num])

                    mix_seed2_randButton = ToolButton(ui.random_symbol, tooltip="Set seed to -1, which will cause a new random number to be used every time")
                    mix_seed2_randButton.click(fn=lambda: mix_seed2_Num.update(value=-1), show_progress=False, inputs=[], outputs=[mix_seed2_Num])

                    mix_seed2_recycleButton = ToolButton(ui.reuse_symbol, tooltip="Reuse seed from last generation")

                mix_psiSlider = gr.Slider(-1,1,
                                step=0.05,
                                value=0.7,
                                label='Truncation (psi)')  
                with gr.Row():
                    mix_maskDrop = gr.Dropdown(
                        choices=[ "total (0xFFFF)", "coarse (0xFF00)", "mid (0x0FF0)", "fine (0x00FF)", "alt1 (0xF0F0)", "alt2 (0x0F0F)", "alt3 (0xF00F)"], label="Interpolation Mask", value="total (0xFFFF)"
                    )
                    mix_mixSlider = gr.Slider(0,1,
                                    step=0.01,
                                    value=0.5,
                                    label='Seed Mix (Crossfade)')

                    def update_mix_range(mask=str, mix_value=float):
                        if "total" in mask:
                            # clamp mix_value
                            if mix_value > 1.0:
                                mix_value = 1.0
                            elif mix_value < -1.0:
                                mix_value = -1.0
                            # rescale to 0-1
                            mix_value = GanGenerator.jmap(mix_value, -1.0, 1.0, 0, 1.0)
                            mix_mixSlider.update(minimum=0, maximum=1, value=mix_value)
                        else:
                            mix_mixSlider.update(minimum=1.5, maximum=1.5, value=mix_value)
                    mix_maskDrop.change(update_mix_range, inputs=[mix_maskDrop, mix_mixSlider], outputs=[], show_progress=False)

                    mix_runButton = gr.Button('Generate Style Mix', variant="primary")

                with gr.Row(elem_id="mix-row"):
                        with gr.Column(elem_classes="mix-item"):
                            mix_seed1_Img = gr.Image(label='Seed 1 Image',sources=['upload','clipboard'], interactive=True, type="filepath", elem_classes="gan-output")
                            mix_seed1_Txt = gr.Markdown(label='Seed 1', value="")
                            mix_seed1_Img.upload(
                                fn=get_seed_from_image,
                                inputs=[mix_seed1_Img],
                                outputs=[mix_seed1_Num],
                                show_progress=False
                            )
                        with gr.Column(elem_classes="mix-item"):
                            mix_styleImg = gr.Image(label='Style Mixed Image', sources=['upload','clipboard'], interactive=True, type="filepath", elem_classes="gan-output")
                            mix_styleImg.upload(
                                fn=get_mix_params_from_image,
                                inputs=[mix_styleImg],
                                outputs=[mix_seed1_Num, mix_seed2_Num, mix_mixSlider, mix_maskDrop],
                                show_progress=False
                            )
                        with gr.Column(elem_classes="mix-item"):
                            mix_seed2_Img = gr.Image(label='Seed 2 Image', sources=['upload','clipboard'], interactive=True, type="filepath", elem_classes="gan-output")
                            mix_seed2_Txt = gr.Markdown(label='Seed 2', value="")
                            mix_seed2_Img.upload(
                                fn=get_seed_from_image,
                                inputs=[mix_seed2_Img],
                                outputs=[mix_seed2_Num],
                                show_progress=False
                            )

        seed_recycleButton.click(fn=copy_seed,show_progress=False,inputs=[seedTxt],outputs=[seedNum])

        simple_runButton.click(fn=model.generate_image_from_ui,
                        inputs=[modelDrop, seedNum, psiSlider],
                        outputs=[resultImg, seedTxt])

        seed1_to_mixButton.click(fn=copy_seed, inputs=[seedTxt],outputs=[mix_seed1_Num])
        seed2_to_mixButton.click(fn=copy_seed, inputs=[seedTxt],outputs=[mix_seed2_Num])

        mix_seed1_recycleButton.click(fn=copy_seed,show_progress=False,inputs=[mix_seed1_Txt],outputs=[mix_seed1_Num])
        mix_seed2_recycleButton.click(fn=copy_seed,show_progress=False,inputs=[mix_seed2_Txt],outputs=[mix_seed2_Num])

        mix_runButton.click(fn=model.generate_mix_from_ui,
                        inputs=[modelDrop, mix_seed1_Num, mix_seed2_Num, mix_psiSlider, mix_maskDrop, mix_mixSlider],
                        outputs=[mix_seed1_Img, mix_seed2_Img, mix_styleImg, mix_seed1_Txt, mix_seed2_Txt])

        return [(ui_component, "GAN Generator", "gan_generator_tab")]



script_callbacks.on_ui_tabs(on_ui_tabs)

def on_ui_settings():
    global_state.init()
    section = ('gan_generator', 'StyleGAN Image Generator')

    shared.opts.add_option('gan_generator_device',
        shared.OptionInfo(default_device(), "Generate using CPU or GPU", gr.Dropdown, {"choices": ["cpu", "cuda", "mps"]}, section=section))
    shared.opts.onchange('gan_generator_device', update_device)

    shared.opts.add_option('gan_generator_image_format',
        shared.OptionInfo("png", "File format for image outputs", gr.Dropdown, {"choices": ["jpg", "png"]}, section=section))
    shared.opts.onchange('gan_generator_image_format', update_image_format)

    shared.opts.add_option('gan_generator_image_pad',
        shared.OptionInfo(1.0, "Image padding factor", gr.Slider, {"minimum":1,"maximum":2,"step":0.05,"info":"Resizes image. If > 1, will pad with black border. Useful for zoomed-in faces."}, section=section))
    shared.opts.onchange('gan_generator_image_pad', update_image_padding)
    
script_callbacks.on_ui_settings(on_ui_settings)

def copy_seed(seedTxt) -> Union[int, None]:
    return str_utils.str2num(seedTxt)

def update_model_list() -> tuple[str]:
    files = file_utils.model_path.glob("*.pkl")
    return [os.path.basename(file) for file in sorted(files, key=lambda file: (os.stat(file).st_mtime, file), reverse=True)]

def default_model() -> Union[str, None]:
    return update_model_list()[0] if update_model_list() else None

def touch_model_file(modelDrop) -> None:
    file_utils.touch( file_utils.model_path / modelDrop )

import torch
def default_device() -> str:
    if torch.backends.mps.is_available():
        default_device = "mps"
    # elif torch.cuda.is_available():
    #     default_device = "cuda"
    else:
        default_device = "cpu"
    return default_device

def update_device():
    global_state.device = shared.opts.data.get('gan_generator_device', default_device())
    if 'cuda' in global_state.device:
        # Ensure cuda_extension binaries are in path
        cuda_extension_path = Path(__file__).resolve().parents[0] / "cuda_extensions_cuda121_py310"
        for plugin in [ "bias_act_plugin", "filtered_lrelu_plugin", "upfirdn2d_plugin"]:
            path = str(cuda_extension_path / plugin)
            if plugin not in sys.path:
                sys.path.append(plugin)
    logger(f"Model: {global_state.device}")

def update_image_format():
    global_state.image_format = shared.opts.data.get('gan_generator_image_format', 'png')
    logger(f"Output format: {global_state.image_format}")

def update_image_padding():
    global_state.image_pad = shared.opts.data.get('gan_generator_image_pad', '1.0')
    logger(f"Output padding: {global_state.image_pad}")


# fetch metadata from drag-and-drop (gr.Image.upload callback)
def get_seed_from_image(img) -> int:
    return get_params_from_image(img)[0]

def get_params_from_image(img) -> tuple[int,float]:
    seed,psi,model_name = -1, 0.7, default_model()
    p = metadata.parse_params_from_image(img)
    seed = p.get('seed',seed)
    seed = p.get('seed', p.get('seed1',seed))
    psi = p.get('psi',psi)
    model_name = p.get('model',model_name)
         
    return seed, psi #, model_name
 
def get_mix_params_from_image(img) -> tuple[int,int,float,str]:
    seed1,seed2,mix = -1, -1, 0
    psi,mask,model_name = 0.7, "total (0xFFFF)", default_model()

    p = metadata.parse_params_from_image(img)
    seed1 = p.get('seed1',seed1)
    seed2 = p.get('seed2',seed2)
    mix = p.get('mix',mix)
    psi = p.get('psi',psi)
    mask = p.get('mask',mask)
    model_name = p.get('model',model_name)

    return seed1, seed2, mix, mask #, model_name
