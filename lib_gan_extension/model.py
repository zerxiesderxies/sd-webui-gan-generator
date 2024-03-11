from __future__ import annotations
# Standard library imports
import os
import pathlib
import pickle
import random
from typing import Union
# Third-party library imports
import numpy as np
import torch
import torch.nn as nn
import torch_utils
import dnnlib
from PIL import Image
# Internal module imports
from modules.images import save_image_with_geninfo
from modules.paths_internal import default_output_dir
from lib_gan_extension import global_state, file_utils, str_utils
class Model:

    @classmethod
    def newSeed(cls) -> int:
        return random.randint(0, 0xFFFFFFFF - 1)

    @classmethod
    def xfade(cls, a,b,x):
        return a*(1.0-x) + b*x # basic linear interpolation

    def __init__(self):
        self.device = None
        self.model_name = None
        self.G = None
        self.outputRoot = pathlib.Path(__file__) / default_output_dir / "stylegan-images"
        file_utils.mkdir_p(self.outputRoot)

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
        file_utils.mkdir_p(self.output_path())

    def output_path(self):
        return self.outputRoot / ".".join(self.model_name.split(".")[:-1])

    def generate_image(self, seed: int, psi: float, save: bool=True) -> np.ndarray:
        filename = f"base-{seed}-{psi}.{global_state.image_format}"
        path = self.output_path() / filename
        if path.exists():
            return Image.open(path)
        else:
            w = self.get_w_from_seed(seed, psi)
            output = self.w_to_img(w)[0]
            self.save_output_to_file(output, filename, params={'seed': seed, 'psi': psi})
            return output

    def save_output_to_file(self, output, filename, params: dict = None):
        path = self.output_path() / filename
        if not path.exists():
            print(f"Generated GAN image with {str(params)}")
            info = {
                'parameters': {
                    'model': self.model_name,
                    **params,
                    'extension': 'gan-generator',
                }
            }
            image = Image.fromarray(output)
            save_image_with_geninfo(image, str(info), str(path))

    def set_model_and_generate_image(self, device: str, model_name: str, seed: int,
                                     psi: float) -> np.ndarray:        
        self.set_device(device)
        self.set_model(model_name)
        if seed == -1:
            seed = self.newSeed()
        seedTxt = 'Seed: ' + str(seed)
        return self.generate_image(seed, psi), seedTxt
        
    def set_model_and_generate_styles(self, device: str, model_name: str, seed1: int, seed2: int,
                                     psi: float, interpType: str, mix: float) -> np.ndarray:
        self.set_device(device)
        self.set_model(model_name)

        if seed1 == -1:
            seed1 = self.newSeed()
        img1 = self.generate_image(seed1, psi)

        if seed2 == -1:
            seed2 = self.newSeed()
        img2 = self.generate_image(seed2, psi)

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

        match interpType:
            case "coarse":
                mask = 0xFF00
            case "mid":
                mask = 0x0FF0
            case "fine":
                mask = 0x00FF
            case "total":
                mask = 0xFFFF
            case _:
                mask = str_utils.str2num(interpType)

        w_base = w_list[0].clone() # transfer onto L image as default

        slider_max = 2.0 # FIXME: this is a hack to fix the slider bug where range is stuck at 0-2
        i = mix / slider_max # rescale between 0 and 1
        if mask != 0xFFFF:
            i = i * 2.0 - 1.0 # rescale between -1 and 1
            if i > 0: # transfer L onto R
                w_base = w_list[1].clone()
            else: # transfer R onto L
                i = abs(i)
                w_list = w_list[::-1] # effectively swap the two seeds
            i *= 1.5 # increase range

        mask = self.num2mask(mask)
        w_base[:,mask,:] = self.xfade(w_list[0][:,mask,:], w_list[1][:,mask,:], i)

        img3 = self.w_to_img(w_base)[0]
        filename = f"mix-{seed1}-{seed2}-{mix}-{interpType}.{global_state.image_format}"
        self.save_output_to_file(img3, filename, params={'seed1': seed1, 'seed2': seed2, 'mix': i, 'mask': interpType})
        
        seedTxt1 = f"Seed 1: {str(seed1)} ({str_utils.num2hex(seed1)})"
        seedTxt2 = f"Seed 2: {str(seed2)} ({str_utils.num2hex(seed2)})"

        return img1, img2, img3, seedTxt1, seedTxt2

    @classmethod
    def weight_vector(cls, size: int, offset:int, total_len:int=16):
        length = size * 2
        cap = int(total_len / 2) - size
        mask = np.array([0] * cap + [1] * length + [0] * cap, dtype=bool)
        mask = np.roll(mask, offset) # shfit by offset
        return mask

    @classmethod
    def mask2num(cls, mask: np.ndarray) -> int:
        int_array = mask.astype(int)
        return int(''.join(map(str, int_array)), 2)

    @classmethod
    def num2mask(cls, num: int) -> np.ndarray:
        return np.array([x=='1' for x in bin(num)[2:].zfill(16)], dtype=bool)

    @classmethod
    def jmap(cls, sourceValue, sourceRangeMin, sourceRangeMax, targetRangeMin, targetRangeMax) -> float:
        if sourceRangeMax == sourceRangeMin:
            raise ValueError("mapping from a range of zero will produce NaN!")
        return targetRangeMin + ((targetRangeMax - targetRangeMin) * (sourceValue - sourceRangeMin)) / (sourceRangeMax - sourceRangeMin)
