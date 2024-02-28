# Simple StyleGAN3 (and StyleGAN2) Generator Extension

Adds a tab to Stable Diffusion Webui that allows the user to generate images from locally downloaded StyleGAN2 or StyleGAN3 models. Created as a proof of concept and is in very early stages. This extension also provides style mixing capabilities. Tested only on Windows.
Based on [NVlabs/stylegan3](https://https://github.com/NVlabs/stylegan3) with style mixing help from [tocantrell/stylegan3-fun-pipeline](https://github.com/tocantrell/stylegan3-fun-pipeline/)

## Features!

### Benefits of StyleGAN 
- Say goodbye to same face; StyleGAN models are capable of generating over 4 billion diverse and highly detailed images.
- Very fast. Can generate new images in tenths of a second on GPU.
- Cohesive and Smooth Latent Space: Can easily interpolate between faces, genders, animals, anything. Want to ask animal ffhq for the exact midpoint between a dog and a cat? StyleGAN can do that!
- Other cool stylegan features like DragGan, Projection of Real Images, 100% Smooth Video Transitions between two images, and more.

### Features of sd-webui-gan-generator Extension
- Simple gui tab for hosting any of your StyleGAN checkpoints and generate images.
- Style mixing between two seeds for even more customization.
- Can easily randomize seed input. Can also let chance decide which seeds to style mix today.
- Can combine this with stable diffusion's inpainting/outpainting or faceswap/IP Adapter to create consistent, realistic, fictional characters.

## Prerequisites (GPU Only)

### Visual Studio Build Tools - WINDOWS

1. Install Visual Studio 2022: This step is required to build some of the dependencies. You can use the Community version of Visual Studio 2022, which can be downloaded from the following link: https://visualstudio.microsoft.com/downloads/
2. OR Install only the VS C++ Build Tools: If you don’t need the full Visual Studio suite, you can choose to install only the VS C++ Build Tools. During the installation process, select the option for “Desktop Development with C++” found under the “Workloads -> Desktop & Mobile” section. The VS C++ Build Tools can be downloaded from this link: https://visualstudio.microsoft.com/visual-cpp-build-tools/

### Building Torch Util Libraries - WINDOWS

1. When running on GPU, the following libraries are built by torch_utils: `bias_act_plugin`, `upfirdn2d_plugin`, and `filtered_lrelu_plugin` (for StyleGAN2 models)
2. The libraries also require `python3XX.lib`. Because sd-webui installs into a venv, you will need to manually copy the .lib yourself, as shown below:
- Open command prompt and type `python`
- `>>> import os, sys`
- `>>> os.path.dirname(sys.executable)`
- Navigate to indicated directory and look for the libs folder. Copy those files and create a folder in the sd-webui environment folder: `stable-diffusion-webui\venv\scripts\libs`
- Alternatively you can add `Python\Python310\libs` to your system path variable

Notes:
- If during run you get a build failed, it mostly will be either due to missing library (e.g. python310.lib) or your cuda toolkit is not compatible with your GPU or pytorch installation.

## Installation

1. Install [AUTOMATIC1111's Stable Diffusion Webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
2. Navigate to the `.\extensions` and tab. Click `Install from URL`
3. Place `https://github.com/zerxiesderxies/sd-webui-gan-generator/` under `URL for extension's git repository` and Install (Note you may need to manually add the directory name `sd-webui-gan-generator`)

## Usage

**NOTE**: StyleGAN2/3 pretrained checkpoints (pickles) contain additional classes (e.g. torch_utils) that are not compatible with stable-diffusion-webui's pickle scanner. You will need to set `--disable-safe-unpickle` in order to load them.  
TODO: Need Workaround  
**WARNING**: Setting `--disable-safe-unpickle` turns off the safe pickle check exposes sd-webui to malicious code hidden in pickle files. Use at your own risk. Please verify the integrity of your .pkl files before using.

### Downloading models

1. Download any StyleGAN2 or StyleGAN3 model you prefer.
- Recommend either `ffhq` or `celeba` pre-trained networks from NVlabs  
[stylegan3 checkpoints here](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan3)  
[stylegan2 checkpoints here](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan2)  
2. Place the checkpoint .pkl files in your `extensions\sd-webui-gan-generator\models` folder

### User Interface

The `Simple Image Gen` tab handles basic seed to image generation.
1. Under model selection, select your model from the drop down menu. Click refresh to update the model list.
2. Under generation device, select whether you want to use your CPU `cpu` or Nvidia GPU `cuda:0`.
3. Simple Image Gen is the basic generation page
- Select your seed (integer from `0 to 2^32-1`)
- Select your truncation psi (`0.7` is a good value to start with). See below for explanation.
- Click `Generate Simple Image` to generate the image. Note this could take some time (slower on CPU). Check command window for status.
- You can also check `Random Seed` for random seed.
- If you are happy with the image, you can send the seed to style mixing for further processing.

#### Style Mixing

The `Style Mixing` tab include simple style mixing features. Style mixing is the process of transferring elements from one image into another. See explanation for further information.
1. Seeds imported from simple gen page, or input your Seed 1 and Seed 2 directly
- Can also click `Pick Seeds For Me` to randomly pick both seed1 and seed2
- Can click `Swap Seeds` to flip the values of Seed 1 and Seed 2.
2. Select your truncation psi and transfer interpolation factor (see below for explanation)
3. Select your method of style transfer in the drop-down menu (see below for explanation)
4. Click `Generate Style Mixing`. The first two seeds and the mixed image should display.

## Parameters Explanation

- **Seed**: Integer input to create the latent vector. Each seed represents an image. Range of 32-bit unsigned integer (`0 to 2^32-1`).
- **Truncation psi**: A float that represents how much to deviate from the global average (1 = no truncation, 0 = global average). Higher values will be more diverse but with worse quality. 
- **Transfer Interpolation Factor**: A float slider used in style mixing that represents how far to settle inside the latent space between two images. A value of 0.0 will be closer to seed 2, and 2.0 will be closer to seed 1.
- **Method of Style Transfer**: The type of style transfer to use. See example image diagram below:

### Methods of Style Transfer
The following methods use the Interpolation Factor.
- **Coarse**: Coarse styles are layers 0-6. Coarse styles govern high-level features such as the subject's pose of in the image or the subject's hair.
- **Fine**: Fine styles are layers 7-14. Fine styles cover the fine details in the image such as color of the eyes or other microstructures.
- I like to think of it like this: If you want to change large aspects of the image's subject, like face shape, hairstyle, but keep the detailed composition like eyebrows, nose, eye colors, and skin tone, use coarse.
- If you want to keep the subject's pose and shape exactly the same but change the background color or filter, use fine.   
The following are independent of the Interpolation Factor.
- **Coarse_Average**: Generates an image with the average midpoint between Seed1 and Seed2 within the coarse layers.
- **Fine_Average**: Generates an image with the average midpoint between Seed1 and Seed2 within the fine layers.
- **Total_Average**: Generates an image with the total midpoint between Seed1 and Seed2 within all layers.

## Credits
Please give credit to me `ZerxiesDerxies` if you plan to start using this.
Credits to `NVLabs` and `tocantrell`. This is my first gradio project and a beginner in ML/AI, so take this with a grain of salt.