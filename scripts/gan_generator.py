import modules.scripts as scripts
import gradio as gr
from glob import glob
from pathlib import Path

from modules import script_callbacks
import json
import re

import gradio as gr
import numpy as np

from scripts.model import Model

import torch
import random

from modules import ui
from modules.ui_components import ToolButton
ui.swap_symbol = "\U00002194"  # ↔️

model = Model()

DESCRIPTION = '''# StyleGAN3 Simple Image Generator Extension

Use this tool to generate random images with a pretrained StyleGAN3 network of your choice. 
Download model pickle files and place them in sd-webui-gan-generator/models folder. 
Supports generation with the cpu or gpu0. See available pretrained networks via [https://github.com/NVlabs/stylegan3](https://github.com/NVlabs/stylegan3).
Recommend using stylegan3-r-ffhq or stylegan2-celebahq
'''

def str2num(string):
    match = re.search(r'(\d+)', string)
    if match:
        number = int(match.group())
        return number
    else:
        return None

def copy_seed(seedTxt):
    number = str2num(seedTxt)
    if number is not None:
        return number

def update_model_list():
    path = Path(__file__).resolve().parents[1] / "models"
    return [Path(file).name for file in glob(str(path / "*.pkl"))]

def default_model():
    return update_model_list()[0] if update_model_list() else None

def default_device():
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
            mix_interpDrop = gr.Dropdown(choices = update_model_list(), value=default_model, label="Model Selection", info="Place into models directory")
            model_refreshButton = ToolButton(value=ui.refresh_symbol, tooltip="Refresh")
            # model_refreshButton.click(fn=lambda: mix_interpDrop.update(choices=update_model_list()), inputs=[],outputs=[mix_interpDrop])
            model_refreshButton.click(fn=lambda: gr.Markdown.update(choices=update_model_list()),outputs=mix_interpDrop)

            deviceDrop = gr.Dropdown(choices = ['cpu','cuda:0','mps'], value=default_device, label='Generation Device', info='Generate using CPU or GPU')
                                
        with gr.Tabs():
            with gr.TabItem('Simple Image Gen'):
                with gr.Row():
                    with gr.Column():
                        psiSlider = gr.Slider(0,
                                        2,
                                        step=0.05,
                                        value=0.7,
                                        label='Truncation (psi)')
                        with gr.Row():
                            seed = gr.Number(label='Seed', value=-1, min_width=150, precision=0)
                            seed_randButton = ToolButton(ui.random_symbol, tooltip="Set seed to -1, which will cause a new random number to be used every time")
                            seed_randButton.click(fn=lambda: seed.update(value=-1), show_progress=False, inputs=[], outputs=[seed])
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
                    mix_seed1 = gr.Number(label='Seed 1', value=-1, min_width=150, precision=0)

                    mix_seed1_recycleButton = ToolButton(ui.reuse_symbol, tooltip="Reuse seed from last generation")
                    mix_seed1_recycleButton.click(fn=copy_seed,show_progress=False,inputs=[seedTxt],outputs=[seed])

                    mix_seed1_randButton = ToolButton(ui.random_symbol, tooltip="Set seed to -1, which will cause a new random number to be used every time")
                    mix_seed1_randButton.click(fn=lambda: mix_seed1.update(value=-1), show_progress=False, inputs=[], outputs=[mix_seed1])

                    mix_seed2 = gr.Number(label='Seed 2', value=-1, min_width=150, precision=0)

                    mix_seed2_recycleButton = ToolButton(ui.reuse_symbol, tooltip="Reuse seed from last generation")
                    mix_seed2_recycleButton.click(fn=copy_seed,show_progress=False,inputs=[seedTxt],outputs=[seed])

                    mix_seed2_randButton = ToolButton(ui.random_symbol, tooltip="Set seed to -1, which will cause a new random number to be used every time")
                    mix_seed2_randButton.click(fn=lambda: mix_seed2.update(value=-1), show_progress=False, inputs=[], outputs=[mix_seed2])

                mix_psiSlider = gr.Slider(0,2,
                                step=0.05,
                                value=0.7,
                                label='Truncation psi')  
                with gr.Row():
                    mix_interp_styleDrop = gr.Dropdown(
                                choices=["coarse", "fine", "total"], label="Interpolation Style", value="coarse"
                                    ),                                   
                    mix_mixSlider = gr.Slider(0,2,
                                    step=0.01,
                                    value=1.0,
                                    label='Seed Mix (Crossfade)')

                    mix_runButton = gr.Button('Generate Style Mix')

                with gr.Row():
                    with gr.Column():
                        seedImg1 = gr.Image(label='Seed 1 Image')
                        seedTxt1 = gr.Markdown(label='Seed 1', value="")
                    with gr.Column():
                        styleImg = gr.Image(label='Style Mixed Image')
                    with gr.Column():
                        seedImg2 = gr.Image(label='Seed 2 Image')
                        seedTxt2 = gr.Markdown(label='Seed 2', value="")

        simple_runButton.click(fn=model.set_model_and_generate_image,
                         inputs=[deviceDrop, mix_interpDrop, seed, psiSlider],
                         outputs=[resultImg, seedTxt])
        seed1_to_mixButton.click(fn=copy_seed, inputs=[seedTxt],outputs=[mix_seed1])
        seed2_to_mixButton.click(fn=copy_seed, inputs=[seedTxt],outputs=[mix_seed2])

        mix_runButton.click(fn=model.set_model_and_generate_styles,
                         inputs=[deviceDrop, mix_interpDrop, mix_seed1, mix_seed2, mix_psiSlider, mix_interp_styleDrop[0], mix_mixSlider],
                         outputs=[seedImg1, seedImg2, styleImg, seedTxt1, seedTxt2])
        seed_recycleButton.click(fn=copy_seed,show_progress=False,inputs=[seedTxt],outputs=[seed])

        return [(ui_component, "GAN Generator", "gan_generator_tab")]

script_callbacks.on_ui_tabs(on_ui_tabs)
