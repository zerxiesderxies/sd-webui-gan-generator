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

    def w_to_image(self, dlatents: Union[List[torch.Tensor], torch.Tensor], noise_mode: str = 'const') -> Image.Image:
        """
        Get an image/np.ndarray from a dlatent W using G and the selected noise_mode. The final shape of the
        returned image will be [len(dlatents), G.img_resolution, G.img_resolution, G.img_channels].
        """
        assert isinstance(dlatents, torch.Tensor), f'dlatents should be a torch.Tensor!: "{type(dlatents)}"'
        if len(dlatents.shape) == 2:
            dlatents = dlatents.unsqueeze(0)  # An individual dlatent => [1, G.mapping.num_ws, G.mapping.w_dim]
        # if not isinstance(dlatents, list):
        #     dlatents = [dlatents] # convert to array if an individual dlatent e.g. [1, G.mapping.num_ws, G.mapping.w_dim]
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

    def find_image_if_exists(self, filename: str) -> Union[None, Image.Image]:
        path = self.output_path() / filename
        if path.exists():
            return Image.open(path)
        return None

    def pad_image(self, image: Image.Image, factor: float=1.0) -> Image.Image:
        resolution = self.G.img_resolution
        new_size = int(resolution*factor)
        padImage = Image.new(image.mode, (new_size, new_size), (0, 0, 0))
        padding = int((new_size-resolution)/2)
        padImage.paste(image, box=(padding, padding))
        return padImage

    def base_image_path(self, seed: int, psi: float) -> str:
        return f"base-{seed}-{psi}.{global_state.image_format}"

    def generate_base_image(self, seed: int, psi: float) -> Image.Image:
        params = {'seed': seed, 'psi': psi}
        w = self.get_w_from_seed(**params)
        img = self.w_to_image(w)
        self.save_image_to_file(img, self.base_image_path(**params), params)
        return img

    def find_or_generate_base_image(self, seed: int, psi: float) -> Image.Image:
        params = {'seed': seed, 'psi': psi}
        log = f"Generated GAN image with {str(params)}"

        output = self.find_image_if_exists(self.base_image_path(**params))
        if output is None:
            output = self.generate_base_image(**params)
        else:
            log += " (cached on disk)"
        print(log)

        return output

    def generate_image(self, seed: int, psi: float, pad: float=1.0) -> Image.Image:
        params = {'seed': seed, 'psi': psi}
        output = self.find_or_generate_base_image(**params)
        if pad != 1.0:
            output = self.pad_image(output, pad)
            padded_path = f"base-{seed}-{psi}-pad{pad}.{global_state.image_format}"
            self.save_image_to_file(output, padded_path, {**params, 'pad': pad})
            # print(f"  Padded image by {pad}x")
        return output

    def save_image_to_file(self, image: Image.Image, filename: str, params: dict = None):
        path = self.output_path() / filename
        info = {
            'parameters': {
                'model': self.model_name,
                **params,
                'extension': 'gan-generator',
            }
        }
        save_image_with_geninfo(image, str(info), str(path))

    def set_model_and_generate_image(self, device: str, model_name: str, seed: int,
                                     psi: float, pad: float) -> np.ndarray:        
        self.set_device(device)
        self.set_model(model_name)
        if seed == -1:
            seed = self.newSeed()
        seedTxt = 'Seed: ' + str(seed)

        return self.generate_image(seed, psi, pad), seedTxt
        
    def set_model_and_generate_styles(self, device: str, model_name: str, seed1: int, seed2: int,
                                     psi: float, interpType: str, mix: float, pad: float) -> np.ndarray:
        self.set_device(device)
        self.set_model(model_name)

        if seed1 == -1:
            seed1 = self.newSeed()
        w1 = self.get_w_from_seed(seed1, psi)
        img1 = self.w_to_image(w1)

        if seed2 == -1:
            seed2 = self.newSeed()
        w2 = self.get_w_from_seed(seed2, psi)
        img2 = self.w_to_image(w2)

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

        w_base = w1.clone() # transfer onto L image as default

        slider_max = 2.0 # FIXME: this is a hack to fix the slider bug where range is stuck at 0-2
        i = mix / slider_max # rescale between 0 and 1
        if mask != 0xFFFF:
            i = i * 2.0 - 1.0 # rescale between -1 and 1
            if i > 0: # transfer L onto R
                w_base = w2.clone()
            else: # transfer R onto L
                i = abs(i)
                w1,w2 = w2,w1 # swap L and R
            i *= 1.5 # increase range

        mask = self.num2mask(mask)
        w_base[:,mask,:] = self.xfade(w1[:,mask,:], w2[:,mask,:], i)

        img3 = self.w_to_image(w_base)
        img3 = self.pad_image(img3,pad)

        filename = f"mix-{seed1}-{seed2}-{mix}-{interpType}-{pad}.{global_state.image_format}"
        self.save_image_to_file(img3, filename, params={'seed1': seed1, 'seed2': seed2, 'mix': mix, 'interp': interpType, 'pad': pad})
        
        seedTxt1 = f"Seed 1: {str(seed1)} ({str_utils.num2hex(seed1)})"
        seedTxt2 = f"Seed 2: {str(seed2)} ({str_utils.num2hex(seed2)})"

        return img1, img2, img3, seedTxt1, seedTxt2

    # experimental function to control weighting vector
    @classmethod
    def weight_vector(cls, width: int, offset:int, total_len:int=16):
        size = width * 2
        cap = int(total_len / 2) - width
        mask = np.array([0] * cap + [1] * size + [0] * cap, dtype=bool)
        mask = np.roll(mask, offset) # shift by offset

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
