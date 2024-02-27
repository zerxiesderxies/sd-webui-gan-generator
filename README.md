# Simple StyleGAN3 (and StyleGAN2) Generator Extension

Adds a tab to the webui that allows the user to generate images from locally downloaded StyleGAN2 or StyleGAN3 models. Created as a proof of concept and is in very early stages. This extension also provides style mixing capabilities. Tested only in Windows.
Based on [NVlabs/stylegan3](https://https://github.com/NVlabs/stylegan3) with style mixing help from [tocantrell/stylegan3-fun-pipeline](https://github.com/tocantrell/stylegan3-fun-pipeline/)

## Features!

### Benefits of StyleGAN 
- Say goodbye to same face; StyleGAN models are capabile of generating over 4 billion unique and highly detailed diverse images
- Very fast. Can generate new images in tenths of a second on GPU for models that are less than 300 MB in size
- Cohesive and Smooth Latent Space: Can easily interpolate between faces, genders, animals, anything. Want to ask animal ffhq for the exact midpoint between a dog and a cat? StyleGAN can do that!
- Other cool stylegan features like DragGan, Projection of Real Images, and more

### Features of Webui Extension
- Simple gui tab for hosting any of your StyleGAN checkpoints
- Style mixing between two seeds for even more customization
- Can easily randomize seed input. Can also let chance decide which seeds to style mix today
- Can combine this with stable diffusion's inpainting/outpainting or faceswap to create consistent, realistic, fictional characters.

## Prerequisites (GPU Only)

### Visual Studio Build Tools - WINDOWS

1. Install Visual Studio 2022: This step is required to build some of the dependencies. You can use the Community version of Visual Studio 2022, which can be downloaded from the following link: https://visualstudio.microsoft.com/downloads/
2. OR Install only the VS C++ Build Tools: If you don’t need the full Visual Studio suite, you can choose to install only the VS C++ Build Tools. During the installation process, select the option for “Desktop Development with C++” found under the “Workloads -> Desktop & Mobile” section. The VS C++ Build Tools can be downloaded from this link: https://visualstudio.microsoft.com/visual-cpp-build-tools/

### Building Torch Util Libraries - WINDOWS

1. When running on GPU, the following libraries are built by torch_utils: `bias_act_plugin`, `upfirdn2d_plugin`, and `filtered_lrelu_plugin` (for StyleGAN2 models)
2. In order to build (in addition to VS Tools), they require python3XX.lib. Because sd-webui installs into a venv, you will need to manually copy your. Alternatively you can add 
- Open command prompt and type `python`
- `>>> import os, sys`
- `>>> os.path.dirname(sys.executable)`
- Navigate to indicated directory and look for libs folder. Copy those files and create a folder in the stable diffusion environment folder: `venv/scripts/libs`
- Alternatively you can add `Python\Python310\libs` to your system path variable

Notes:
- If during run you get a build failed, it mostly will be either due to missing library (e.g. python310.lib) or your cuda toolkit is not compatible with your GPU or pytorch installation.

## Installation

1. Install [AUTOMATIC1111's Stable Diffusion Webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
2. Navigate to the `.\extensions` and tab. Click `Install from URL`
3. Place `https://github.com/zerxiesderxies/sd-webui-gan-generator/` under `URL for extension's git repository` and Install (Note you may need to manually add the directory name `sd-webui-gan-generator`)

## Usage

**NOTE**: StyleGAN2 pretrained checkpoints (pickles) contain additional classes (e.g. torch_utils) that are not compatible with Stable Diffusions pickle scanner. You will need to set `--disable-safe-unpickle` in order to load them. TODO: Need Workaround
**WARNING**: Setting `--disable-safe-unpickle` exposes your webui to malicious code hidden in pickle files. Use at your own risk. Please verify the integrity of your .pkl files before using.

### Downloading models

1. Download any StyleGAN2 or StyleGAN3 model you prefer.
- Recommend either ffhq or celeba pre-trained networks from NVlabs [stylegan3 models](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan3) [stylegan2 models](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan2)
2. Place checkpoint .pkl files in your `extensions\sd-webui-gan-generator\models` folder

### UI

The basic UI falls under `Simple Image Gen` tab and handles seed to image generation
1. Under model selection, select your model from the drop down menu. Click refresh to update the model list
2. Under generation device, select whether you want to use CPU or Nvidia GPU (cuda:0)
3. Simple Image Gen is the basic generation page
- Select your seed (integer from `0 to 2^32-1`)
- Select your truncation psi (`0.7` is a good value to start with). See below for explanation.
- Click `Generate Simple Image` to generate the image. Note this could take some time (slower on CPU). Check command window for status
- You can also check `Random Seed` for random seed
- If you are happy with the seed, you can send the seed to style mixing for further processing

#### Style Mixing

Simple Style Mixing features are included under the `Style Mixing` tab. Style mixing is the process of transferring elements from one image into another. See explanation for further information.
1. Seeds imported from simple gen page, or input your Seed 1 and Seed 2 directly
- Can also click `Pick Seeds For Me` to randomly pick both seed1 and seed2
- Can click `Swap Seeds` to flip the values of Seed 1 and Seed 2.
2. Select your truncation psi and transfer interpolation factor (see below for explanation)
3. Select your method of style transfer in the drop-down menu (see below for explanation)
4. Click `Generate Style Mixing`. The first two seeds and the mixed image should display.

## Parameters Explanation

- **Seed**: The input to create the latent vector. Each seed represents an image.
- **Truncation psi**: How much to deviate from the average (1 = no truncation, 0 = average). Higher values will be more diverse but with worse quality.
- **Transfer Interpolation Factor**: A slider
- **Method of Style Transfer**: The type of style transfer to use. See below:

### Methods of Style Transfer
The following use the Interpolation Factor
- **Coarse**:
- **Fine**:
The following are independent of the Interpolation Factor
- **Coarse_Average**:
- **Fine_Average**:
- **Total_Average**:

## Credits
Please give credit to me `ZerxiesDerxies` if you plan to start using this.
Credits to NVLabs and tocantrell. This is my first gradio project so apologies in advance for the mess.