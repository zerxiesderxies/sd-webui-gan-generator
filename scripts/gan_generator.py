# Standard library imports
import json
import re
import os
import platform
import subprocess as sp
import glob
from pathlib import Path
import random
# Third-party library imports
import numpy as np
import torch
from typing import Union
import gradio as gr
# Internal module imports
from modules import script_callbacks, shared, ui_components, scripts, ui
from modules.ui_components import ToolButton
from scripts.model import Model, newSeed
import scripts.global_state as global_state

ui.swap_symbol = "\U00002194"  # ↔️
ui.lucky_symbol = "\U0001F340"  # 🍀
ui.folder_symbol = "\U0001F4C1"  # 📁

model = Model()

DESCRIPTION = '''# StyleGAN Image Generator Extension

Use this tool to generate random images with a pretrained StyleGAN3 network of your choice. 
Download model pickle files and place them in sd-webui-gan-generator/models folder. 
Supports generation with the cpu or gpu0. See available pretrained networks via [https://github.com/NVlabs/stylegan3](https://github.com/NVlabs/stylegan3).
Recommend using stylegan3-r-ffhq or stylegan2-celebahq
'''

def str2num(string) -> Union[int, None]:
    match = re.search(r'(\d+)$', string)
    return int(match.group()) if match else None

def copy_seed(seedTxt) -> Union[int, None]:
    return str2num(seedTxt)

model_path = Path(__file__).resolve().parents[1] / "models"

def update_model_list() -> tuple[str]:
    files = glob.glob(str(model_path / "*.pkl"))
    return [Path(file).name for file in sorted(files, key=lambda file: (Path(file).stat().st_mtime, file), reverse=True)]

def default_model() -> Union[str, None]:
    return update_model_list()[0] if update_model_list() else None

def touch_model_file(modelDrop) -> None:
    filename = str(model_path / modelDrop)
    with open(filename, 'a'):
        os.utime(filename, None)  # Update the modification timestamp

def default_device() -> str:
    if torch.backends.mps.is_available():
        default_device = "mps"
    elif torch.cuda.is_available():
        default_device = "cuda:0"
    else:
        default_device = "cpu"
    return default_device

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False, css='style.css') as ui_component:
        gr.Markdown(DESCRIPTION)
        with gr.Row():
            modelDrop = gr.Dropdown(choices = update_model_list(), value=default_model, label="Model Selection", info="Place into models directory", elem_id="models")
            modelDrop.input(fn=touch_model_file, inputs=[modelDrop], outputs=[])
 
            model_refreshButton = ToolButton(value=ui.refresh_symbol, tooltip="Refresh")
            model_refreshButton.click(fn=lambda: gr.Dropdown.update(choices=update_model_list()),outputs=modelDrop)
 
            deviceDrop = gr.Dropdown(choices = ['cpu','cuda:0','mps'], value=default_device, label='Generation Device', info='Generate using CPU or GPU', elem_id="device")
 
            with gr.Group():
                with gr.Column():
                    gr.Markdown(label='Output Folder', value="Output folder", elem_id="output-folder")
                    folderButton = ToolButton(ui.folder_symbol, visible=not shared.cmd_opts.hide_ui_dir_config, tooltip="Open image output directory", elem_id="open-folder")
                    folderButton.click(
                        fn=lambda images, index: open_folder(model.outputRoot),
                        inputs=[],
                        outputs=[],
                    )

        with gr.Tabs():
            with gr.TabItem('Simple Image Gen'):
                with gr.Row():
                    with gr.Column():
                        psiSlider = gr.Slider(0,2,
                                        step=0.05,
                                        value=0.7,
                                        label='Truncation (psi)')
                        with gr.Row():
                            seedNum = gr.Number(label='Seed', value=-1, min_width=150, precision=0)

                            seed_randButton = ToolButton(ui.random_symbol, tooltip="Set seed to -1, which will cause a new random number to be used every time")
                            seed_randButton.click(fn=lambda: seedNum.update(value=-1), show_progress=False, inputs=[], outputs=[seedNum])

                            seed_recycleButton = ToolButton(ui.reuse_symbol, tooltip="Reuse seed from last generation")
 
                        simple_runButton = gr.Button('Generate Simple Image')

                    with gr.Column():
                        resultImg = gr.Image(label='Result', elem_id='result')
                        seedTxt = gr.Markdown(label='Output Seed')
                        with gr.Row():
                            seed1_to_mixButton = gr.Button('Send to Seed Mixer › Left')
                            seed2_to_mixButton = gr.Button('Send to Seed Mixer › Right')

            with gr.TabItem('Seed Mixer'):
                with gr.Row():
                    mix_seed1_Num = gr.Number(label='Seed 1', value=-1, min_width=150, precision=0)

                    mix_seed1_luckyButton = ToolButton(ui.lucky_symbol, tooltip="Roll generate a new seed")
                    mix_seed1_luckyButton.click(fn=lambda: mix_seed1_Num.update(value=newSeed()), show_progress=False, inputs=[], outputs=[mix_seed1_Num])

                    mix_seed1_randButton = ToolButton(ui.random_symbol, tooltip="Set seed to -1, which will cause a new random number to be used every time")
                    mix_seed1_randButton.click(fn=lambda: mix_seed1_Num.update(value=-1), show_progress=False, inputs=[], outputs=[mix_seed1_Num])

                    mix_seed1_recycleButton = ToolButton(ui.reuse_symbol, tooltip="Reuse seed from last generation")

                    mix_seed2_Num = gr.Number(label='Seed 2', value=-1, min_width=150, precision=0)

                    mix_seed2_luckyButton = ToolButton(ui.lucky_symbol, tooltip="Roll generate a new seed")
                    mix_seed2_luckyButton.click(fn=lambda: mix_seed2_Num.update(value=newSeed()), show_progress=False, inputs=[], outputs=[mix_seed2_Num])

                    mix_seed2_randButton = ToolButton(ui.random_symbol, tooltip="Set seed to -1, which will cause a new random number to be used every time")
                    mix_seed2_randButton.click(fn=lambda: mix_seed2_Num.update(value=-1), show_progress=False, inputs=[], outputs=[mix_seed2_Num])

                    mix_seed2_recycleButton = ToolButton(ui.reuse_symbol, tooltip="Reuse seed from last generation")

                mix_psiSlider = gr.Slider(0,2,
                                step=0.05,
                                value=0.7,
                                label='Truncation (psi)')  
                with gr.Row():
                    mix_interp_styleDrop = gr.Dropdown(
                        choices=["coarse", "fine", "total"], label="Interpolation Style", value="coarse"
                    )
                    mix_mixSlider = gr.Slider(0,2,
                                    step=0.01,
                                    value=1.0,
                                    label='Seed Mix (Crossfade)')

                    mix_runButton = gr.Button('Generate Style Mix')

                with gr.Row():
                    with gr.Column():
                        mix_seed1_Img = gr.Image(label='Seed 1 Image')
                        mix_seed1_Txt = gr.Markdown(label='Seed 1', value="")
                    with gr.Column():
                        mix_styleImg = gr.Image(label='Style Mixed Image')
                    with gr.Column():
                        mix_seed2_Img = gr.Image(label='Seed 2 Image')
                        mix_seed2_Txt = gr.Markdown(label='Seed 2', value="")

        seed_recycleButton.click(fn=copy_seed,show_progress=False,inputs=[seedTxt],outputs=[seedNum])

        simple_runButton.click(fn=model.set_model_and_generate_image,
                         inputs=[deviceDrop, modelDrop, seedNum, psiSlider],
                         outputs=[resultImg, seedTxt])

        seed1_to_mixButton.click(fn=copy_seed, inputs=[seedTxt],outputs=[mix_seed1_Num])
        seed2_to_mixButton.click(fn=copy_seed, inputs=[seedTxt],outputs=[mix_seed2_Num])

        mix_seed1_recycleButton.click(fn=copy_seed,show_progress=False,inputs=[mix_seed1_Txt],outputs=[mix_seed1_Num])
        mix_seed2_recycleButton.click(fn=copy_seed,show_progress=False,inputs=[mix_seed2_Txt],outputs=[mix_seed2_Num])

        mix_runButton.click(fn=model.set_model_and_generate_styles,
                         inputs=[deviceDrop, modelDrop, mix_seed1_Num, mix_seed2_Num, mix_psiSlider, mix_interp_styleDrop, mix_mixSlider],
                         outputs=[mix_seed1_Img, mix_seed2_Img, mix_styleImg, mix_seed1_Txt, mix_seed2_Txt])

        return [(ui_component, "GAN Generator", "gan_generator_tab")]

script_callbacks.on_ui_tabs(on_ui_tabs)

def on_ui_settings():
    global_state.init()
    section = ('gan_generator', 'StyleGAN Image Generator')
    shared.opts.add_option('gan_generator_image_format',
        shared.OptionInfo("jpg", "File format for image outputs", gr.Dropdown, {"choices": ["jpg", "png"]}, section=section))
    shared.opts.onchange('gan_generator_image_format', update_image_format)
    
script_callbacks.on_ui_settings(on_ui_settings)

def update_image_format():
    global_state.image_format = shared.opts.data.get('gan_generator_image_format', 'jpg')
    print(f"GAN Output Image Format: {global_state.image_format}")

def open_folder(f, images=None, index=None):
    if shared.cmd_opts.hide_ui_dir_config:
        return

    try:
        if 'Sub' in shared.opts.open_dir_button_choice:
            image_dir = os.path.split(images[index]["name"].rsplit('?', 1)[0])[0]
            if 'temp' in shared.opts.open_dir_button_choice or not ui_tempdir.is_gradio_temp_path(image_dir):
                f = image_dir
    except Exception:
        pass

    if not os.path.exists(f):
        msg = f'Folder "{f}" does not exist. After you create an image, the folder will be created.'
        print(msg)
        gr.Info(msg)
        return
    elif not os.path.isdir(f):
        msg = f"""
WARNING
An open_folder request was made with an argument that is not a folder.
This could be an error or a malicious attempt to run code on your computer.
Requested path was: {f}
"""
        print(msg, file=sys.stderr)
        gr.Warning(msg)
        return

    path = os.path.normpath(f)
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        sp.Popen(["open", path])
    elif "microsoft-standard-WSL2" in platform.uname().release:
        sp.Popen(["wsl-open", path])
    else:
        sp.Popen(["xdg-open", path])
