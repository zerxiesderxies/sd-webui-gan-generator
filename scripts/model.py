from __future__ import annotations

import os
import pathlib
import pickle
# import sys
import random

import numpy as np
import torch
import torch.nn as nn
import torch_utils
import dnnlib

from modules.images import save_image_with_geninfo
from modules.paths_internal import default_output_dir
from PIL import Image

def xfade(a,b,x):
    return a*(1.0-x) + b*x

def mkdir_p(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        # If the directory already exists, it's okay
        pass
    except OSError as e:
        # Handle other errors
        print(f"Error creating directory: {e}")
class Model:
    def __init__(self):
        self.device = None
        self.model_name = None
        self.G = None

    def _load_model(self, model_name: str) -> nn.Module:
        path = pathlib.Path(__file__).resolve().parents[1] / 'models' / model_name 
        
        # WARNING: Verify StyleGAN3 checkpoints before loading.
        # Safety check needs to be disabled because required classes
        # in StyleGAN3 (e.g. torch_utils) are not included in 
        # sd-webui approved class list. Use of this extension is
        # at your own risk.
        
        with open(path, 'rb') as f:
            G = pickle.load(f)['G_ema']
        G.eval()
        G.to(self.device)
        return G


    def w_to_img(self, dlatents: Union[List[torch.Tensor], torch.Tensor], noise_mode: str = 'const') -> np.ndarray:
        """
        Get an image/np.ndarray from a dlatent W using G and the selected noise_mode. The final shape of the
        returned image will be [len(dlatents), G.img_resolution, G.img_resolution, G.img_channels].
        """
        assert isinstance(dlatents, torch.Tensor), f'dlatents should be a torch.Tensor!: "{type(dlatents)}"'
        if len(dlatents.shape) == 2:
            dlatents = dlatents.unsqueeze(0)  # An individual dlatent => [1, G.mapping.num_ws, G.mapping.w_dim]
        try:
            synth_image = self.G.synthesis(dlatents, noise_mode=noise_mode)
        except:
            synth_image = self.G.synthesis(dlatents, noise_mode=noise_mode, force_fp32=True)
        
        synth_image = (synth_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return synth_image.cpu().numpy()

    def random_z_dim(self, seed) -> np.ndarray:
        z = np.random.RandomState(seed).randn(1, self.G.z_dim) 
        if self.device == 'mps':
            z = torch.tensor(z).float().cpu().numpy() # convert to float32 for mac
        return z

    def get_w_from_seed(self, seed: int, psi: float) -> torch.Tensor:
        """Get the dlatent from a random seed, using the truncation trick (this could be optional)"""
        z = self.random_z_dim(seed)
        w = self.G.mapping(torch.from_numpy(z).to(self.device), None)
        w_avg = self.G.mapping.w_avg
        w = w_avg + (w - w_avg) * psi

        return w

    def get_w_from_mean_z(self, psi: float) -> torch.Tensor:
        """Get the dlatent from the mean z space"""
        w = self.G.mapping(torch.zeros((1, self.G.z_dim)).to(self.device), None)
        w_avg = self.G.mapping.w_avg
        w = w_avg + (w - w_avg) * psi

        return w

    def get_w_from_mean_w(self, seed: int, psi: float) -> torch.Tensor:
        """Get the dlatent of the mean w space"""
        w = self.G.mapping.w_avg.unsqueeze(0).unsqueeze(0).repeat(1, 16, 1).to(self.device)
        return w
    
    def set_device(self, device='cpu') -> None:
        if (device == self.device):
            return
        self.device = device
        self.G = None
    
    def set_model(self, model_name: str) -> None:
        if model_name == self.model_name and self.G is not None:
            return
        self.model_name = model_name
        self.G = self._load_model(model_name)

    def generate_image(self, seed: int, psi: float, save: bool=True) -> np.ndarray:
        w = self.get_w_from_seed(seed, psi)
        output = self.w_to_img(w)[0]
        info = { 'GAN-generator': {'seed': seed, 'psi': psi, 'model': self.model_name} }
        filename = f"{self.model_name.replace('.pkl', '')}-{seed}-{psi}.jpg"
        # filename = f"{seed}-{psi}.jpg"
        # filename = os.path.join(default_output_dir(), "gan", self.model_name, filename)
        path = os.path.join(default_output_dir, "gan-images")
        mkdir_p(path)
        filename = os.path.join(path, filename)
        if not os.path.exists(filename):
            image = Image.fromarray(output)
            save_image_with_geninfo(image, str(info), filename )
        return output

    def set_model_and_generate_image(self, device: str, model_name: str, seed: int,
                                     psi: float) -> np.ndarray:        
        self.set_device(device)
        self.set_model(model_name)
        if seed == -1:
            seed = random.randint(0, 0xFFFFFFFF - 1)        
        outputSeedStr = 'Seed: ' + str(seed)
        print(f"Generating GAN image with {{ seed: {seed}, psi: {psi} }}")
        return self.generate_image(seed, psi), outputSeedStr
        
    def set_model_and_generate_styles(self, device: str, model_name: str, seed1: int, seed2: int,
                                     psi: float, styleDrop: str, style_interp: float) -> np.ndarray:
        self.set_device(device)
        self.set_model(model_name)
        im1 = self.generate_image(seed1, psi)
        im2 = self.generate_image(seed2, psi)
        w_avg = self.G.mapping.w_avg
        w_list = []

        z = self.random_z_dim(seed1)
        w = self.G.mapping(torch.from_numpy(z).to(self.device), None)
        w = w_avg + (w - w_avg) * psi
        w_list.append(w)
        
        z = self.random_z_dim(seed2)
        w = self.G.mapping(torch.from_numpy(z).to(self.device), None)
        w = w_avg + (w - w_avg) * psi
        w_list.append(w)


        if styleDrop == "total":
            i = style_interp / 2.0  # scaled between 0 and 1
            w_base = xfade(w_list[0], w_list[1], i)
        else:
            i = style_interp # * 2.0 # input should be btwn 0 and 1, then we multiply by 2 to fit these calculations
            if i > 1.0: # mirror across middle
                w_list = w_list[::-1] # effectively swap the two seeds
                i = 2.0 - i
            w_base = w_list[0].clone()
            if styleDrop == "fine":
                w_base[:,8:,:] = xfade(w_base[:,8:,:], w_list[1][:,8:,:], i)
            elif styleDrop == "coarse":
                w_base[:,:7,:] = xfade(w_base[:,:7,:], w_list[1][:,:7,:], i)

        # print(f"mixing w/ style: {styleDrop}, i: {i}")
     
        im3 = self.w_to_img(w_base)[0]
        
        seed1txt =  'Seed 1: ' + str(seed1)
        seed2txt =  'Seed 2: ' + str(seed2)

        return im1, im2, im3, seed1txt, seed2txt
        
