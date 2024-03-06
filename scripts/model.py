from __future__ import annotations

import os
import pathlib
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import torch_utils
import dnnlib

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

    def get_w_from_seed(self, seed: int, truncation_psi: float) -> torch.Tensor:
        """Get the dlatent from a random seed, using the truncation trick (this could be optional)"""
        z = np.random.RandomState(seed).randn(1, self.G.z_dim)
        w = self.G.mapping(torch.from_numpy(z).to(self.device), None)
        w_avg = self.G.mapping.w_avg
        w = w_avg + (w - w_avg) * truncation_psi

        return w

    def get_w_from_mean_z(self, truncation_psi: float) -> torch.Tensor:
        """Get the dlatent from the mean z space"""
        w = self.G.mapping(torch.zeros((1, self.G.z_dim)).to(self.device), None)
        w_avg = self.G.mapping.w_avg
        w = w_avg + (w - w_avg) * truncation_psi

        return w

    def get_w_from_mean_w(self, seed: int, truncation_psi: float) -> torch.Tensor:
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

    def generate_image(self, seed: int, truncation_psi: float) -> np.ndarray:
        w = self.get_w_from_seed(seed, truncation_psi)
        return self.w_to_img(w)[0]


    def set_model_and_generate_image(self, device: str, model_name: str, seed: int,
                                     truncation_psi: float, randomSeed: bool) -> np.ndarray:
        
        self.set_device(device)
        self.set_model(model_name)
        if randomSeed:
            import random
            seed = random.randint(0, 4294967295 - 1)        
        outputSeedStr = 'Seed: ' + str(seed)
        return self.generate_image(seed, truncation_psi), outputSeedStr, seed
        
    def set_model_and_generate_styles(self, device: str, model_name: str, seed1: int, seed2: int,
                                     truncation_psi: float, styleDrop: str, style_interp : float) -> np.ndarray:        
        self.set_device(device)
        self.set_model(model_name)
        im1 = self.generate_image(seed1, truncation_psi)
        im2 = self.generate_image(seed2, truncation_psi)
        
        w_avg = self.G.mapping.w_avg
        w_list = []

        z = np.random.RandomState(seed1).randn(1, self.G.z_dim)
        w = self.G.mapping(torch.from_numpy(z).to(self.device), None)
        w = w_avg + (w - w_avg) * truncation_psi
        w_list.append(w)
        
        z = np.random.RandomState(seed2).randn(1, self.G.z_dim)
        w = self.G.mapping(torch.from_numpy(z).to(self.device), None)
        w = w_avg + (w - w_avg) * truncation_psi
        w_list.append(w)

        w_base = w_list[0].clone()
        # Using coarse method to transfer styles
        #w_base[:,:7,:] = w_list[1][:,:7,:]
        if styleDrop == "fine":
            w_base[:,8:,:] = w_list[1][:,8:,:]+w_base[:,8:,:]*style_interp
        elif styleDrop == "coarse":
            w_base[:,:7,:] = w_list[1][:,:7,:]+w_base[:,:7,:]*style_interp
        elif styleDrop == "coarse_average":
            w_base[:,:7,:] = (w_list[1][:,:7,:]+w_base[:,:7,:])*0.5
        elif styleDrop == "fine_average":
            w_base[:,8:,:] = (w_list[1][:,8:,:]+w_base[:,8:,:])*0.5
        elif styleDrop == "total_average":
            w_base = (w_list[0] + w_list[1])*0.5

     
        im3 = self.w_to_img(w_base)[0]
        
        seed1txt =  'Seed 1: ' + str(seed1)
        seed2txt =  'Seed 2: ' + str(seed2)

        return im1, im2, im3, seed1txt, seed2txt
        
