# Simple StyleGAN3 (and StyleGAN2) Generator Extension

Adds a tab to Stable Diffusion Webui that allows the user to generate images from locally downloaded StyleGAN2 or StyleGAN3 models. Created as a proof of concept and is in very early stages. This extension also provides style mixing capabilities.  
NOTE: Tested only on Windows.  
Based on [NVlabs/stylegan3](https://github.com/NVlabs/stylegan3) with style mixing help from [tocantrell/stylegan3-fun-pipeline](https://github.com/tocantrell/stylegan3-fun-pipeline/)

## Features!

### Benefits of StyleGAN

- Say goodbye to the same face; StyleGAN models are capable of generating over 4 billion diverse and highly detailed images!
- It's very fast, generating new images in tenths of a second on GPU.
- The cohesive and smooth latent space can be easily interpolated between faces, genders, animals, anything. Want to ask animal ffhq for the exact midpoint between a dog and a cat? StyleGAN can do that!
- Other cool stylegan features like DragGan, Projection of Real Images, 100% Smooth Video Transitions between two images, and more.

### Features of sd-webui-gan-generator Extension

- A simple GUI tab for generating images from any StyleGAN checkpoint.
- Style mixing between two seeds for even more customization.
- Easy randomization of seed input -- let chance decide which seeds to style mix today.
- Most usefully, it can be combined with stable diffusion's inpainting/outpainting or faceswap/IP Adapter to create consistent, realistic, fictional characters!

## Prerequisites (GPU Only)

### Visual Studio Build Tools - WINDOWS

1. Install Visual Studio 2022: This step is required to build some of the dependencies. You can use the Community version of Visual Studio 2022, which can be downloaded from the following link: https://visualstudio.microsoft.com/downloads/
2. OR Install only the VS C++ Build Tools: If you don’t need the full Visual Studio suite, you can choose to install only the VS C++ Build Tools. During the installation process, select the option for “Desktop Development with C++” found under the “Workloads -> Desktop & Mobile” section. The VS C++ Build Tools can be downloaded from this link: https://visualstudio.microsoft.com/visual-cpp-build-tools/

### Building Torch Util Libraries - WINDOWS

1. Note: When running on GPU, the following libraries are built by torch_utils: `bias_act_plugin`, `upfirdn2d_plugin`, and `filtered_lrelu_plugin` (for StyleGAN2 models)
2. The libraries require `python3XX.lib`. Because sd-webui installs into a venv, you will need to go to a locally installed Python similar to the version in sd-webui (e.g. 3.10) and manually copy the .lib yourself, as shown below:

- Open command prompt and type `python`
- `>>> import os, sys`
- `>>> os.path.dirname(sys.executable)`
- Navigate to indicated directory and look for the libs folder. Copy those files and create a libs folder in the sd-webui environment folder: `stable-diffusion-webui\venv\scripts\libs`
- Alternatively you can add `Python\Python310\libs` to your system path variable

Notes:

- If you get failed build, it mostly will be either due to missing library (e.g. python310.lib) or your cuda toolkit is not compatible with your GPU or pytorch installation.

## Installation

1. Install [AUTOMATIC1111's Stable Diffusion Webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
2. Navigate to the `.\extensions` and click `Install from URL`
3. Place `https://github.com/zerxiesderxies/sd-webui-gan-generator/` under `URL for extension's git repository` and Install  
   (Note you may need to manually add the directory name `sd-webui-gan-generator`)

## Usage

**NOTE**: StyleGAN2/3 pretrained checkpoints (pickles) contain additional classes (e.g. torch_utils) that are not compatible with stable-diffusion-webui's pickle scanner. You will need to set `--disable-safe-unpickle` in order to load them.  
TODO: Need Workaround  
**WARNING**: Setting `--disable-safe-unpickle` turns off the safe pickle check and exposes sd-webui to malicious code hidden in pickle files. Use at your own risk. Please verify the integrity of your .pkl files before using.

### Downloading models

1. Download any StyleGAN2 or StyleGAN3 model you prefer.

- Recommend either `ffhq` or `celeba` pre-trained networks from NVlabs  
  [stylegan3 checkpoints here](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan3)  
  [stylegan2 checkpoints here](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan2)

2. Place the checkpoint .pkl files in your `extensions\sd-webui-gan-generator\models` folder

### User Interface

![MainPage](https://github.com/zerxiesderxies/sd-webui-gan-generator/assets/161509935/8143466c-3861-4535-b01b-eb3bf62eba98)  
The `Simple Image Gen` tab handles basic seed to image generation.

1. Under model selection, select your model from the drop down menu. Click refresh to update the model list.
2. Under generation device, select whether you want to use your CPU `cpu`, Nvidia GPU `cuda:0`, or Mac GPU `mps`.
3. How to Generate an Image:

- Select a seed (integer from `0 to 2^32-1`)
- Select your truncation psi (`0.7` is a good value to start with).
- Click `Generate Simple Image` to generate the image. Note: this could take some time on a slower CPU. Check the command window for status.
- Hit the dice button for random seed (-1 value), and the recycle button to replay the last seed.
- If you are happy with the image, you can send the seed to the Style Mixer tab for further processing.

#### Style Mixing

![StyleMixing](https://github.com/zerxiesderxies/sd-webui-gan-generator/assets/161509935/b934563f-dccf-4a28-b111-fe92a480f41b)  
The `Style Mixing` tab include simple style mixing features. Style mixing is the process of transferring elements from one image into another. See explanation for further information.

1. Seeds imported from simple gen page, or input your Seed 1 and Seed 2 directly

- Can also click `Pick Seeds For Me` to randomly pick both seed1 and seed2
- Can click `Swap Seeds` to flip the values of Seed 1 and Seed 2.

2. Select your truncation psi and transfer interpolation factor.
3. Select your method of style transfer in the drop-down menu.
4. Click `Generate Style Mixing`. You should see three images being generated: the first two seeds and the mixed image.

## Explanation of the Parameters

- **Seed**: Integer input to create the latent vector. Each seed represents an image. Range of 32-bit unsigned integer (`0 to 2^32-1`).
- **Truncation psi**: A float that represents how much to deviate from the global average (1 = no truncation, 0 = global average). Higher values will be more diverse but with worse quality. Default is 0.7.
- **Cross-fade Interpolation**: A float slider for mixing the latent space between the two images.
- **Transfer Method**:

  - **Coarse**: This modifies layers 0-6, which govern high-level features. Use this if you want to modify pose, face shape, or hair style, but keep the compositional details of eyebrows, nose, eye color, and skin tone.
  - **Fine**: This modifies layers 7-14. Use this if you want to keep the subject's pose and shape but modify fine details such as
    eyebrows, nose, eye color, and skin tone.
  - **Total**: This mode interpolates across all layers, creating a truly linear blend between the images.

## Credits

Please give credit to `ZerxiesDerxies` if you end up using this.
Credits to `NVLabs` and `tocantrell`. This is my first gradio project and a beginner in ML/AI, so take this with a grain of salt.
Mac OS support, interpolation fix, and GUI enhancements by `dfl`
