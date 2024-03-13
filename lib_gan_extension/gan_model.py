from __future__ import annotations
from typing import Union

import torch
import torch.nn as nn
import torch_utils
import dnnlib
import pickle

import numpy as np
from PIL import Image

class GanModel:
    def __init__(self, model: str, device: str='cpu'):
        # WARNING: Verify StyleGAN3 checkpoints before loading.
        # Safety check needs to be disabled because required classes
        # in StyleGAN3 (e.g. torch_utils) are not included in 
        # sd-webui approved class list. Use of this extension is
        # at your own risk.
        with open(model, 'rb') as f:
            self.G = pickle.load(f)['G_ema']
        self.G.eval()
        self.set_device(device)

    def set_device(self, device: str):
        self.device = device
        self.G.to(device)

    @property
    def img_resolution(self) -> int:
        return self.G.img_resolution

    def w_to_image(self, dlatents: Union[List[torch.Tensor], torch.Tensor], noise_mode: str = 'const') -> Image.Image:
        """
        Get an image/np.ndarray from a dlatent W using G and the selected noise_mode. The final shape of the
        returned image will be [len(dlatents), G.img_resolution, G.img_resolution, G.img_channels].
        """
        assert isinstance(dlatents, torch.Tensor), f'dlatents should be a torch.Tensor!: "{type(dlatents)}"'
        if len(dlatents.shape) == 2:
            dlatents = dlatents.unsqueeze(0)  # An individual dlatent => [1, G.mapping.num_ws, G.mapping.w_dim]
        try:
            img = self.G.synthesis(dlatents, noise_mode=noise_mode)
        except:
            img = self.G.synthesis(dlatents, noise_mode=noise_mode, force_fp32=True)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        img = img.cpu().numpy()
        images = [Image.fromarray(i) for i in img]
        return images[0] if len(images) == 1 else images

    def random_z_dim(self, seed: int) -> np.ndarray:
        return np.random.RandomState(seed).randn(1, self.G.z_dim).astype(np.float32)

    def get_w_from_seed(self, seed: int, psi: float) -> torch.Tensor:
        """Get the dlatent from a random seed, using the truncation trick (this could be optional)"""
        z = torch.from_numpy( self.random_z_dim(seed) ).to(self.device)
        w = self.G.mapping(z, None)
        return self.blend_w_with_mean(w, psi)

    def get_w_from_mean_z(self, psi: float) -> torch.Tensor:
        """Get the dlatent from the mean z space"""
        w = self.G.mapping(torch.zeros((1, self.G.z_dim)).to(self.device), None)
        return self.blend_w_with_mean(w, psi)

    def get_w_from_mean_w(self) -> torch.Tensor:
        """Get the dlatent of the mean w space"""
        # how is this different than self.get_w_from_mean_z(0) ?
        w = self.G.mapping.w_avg.unsqueeze(0).unsqueeze(0).repeat(1, 16, 1).to(self.device)
        return w

    def blend_w_with_mean(self, w: torch.Tensor, psi: float=0.7) -> torch.Tensor:
        w_avg = self.G.mapping.w_avg
        return w_avg + (w - w_avg) * psi

