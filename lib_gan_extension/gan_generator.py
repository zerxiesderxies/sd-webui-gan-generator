from __future__ import annotations
from typing import Union
from pathlib import Path
import numpy as np
from PIL import Image
import random
import sys

from modules.images import save_image_with_geninfo
from modules.paths_internal import default_output_dir

from lib_gan_extension import GanModel, global_state, file_utils, str_utils, metadata
from .global_state import logger

class GanGenerator:
    def __init__(self):
        self.device = None
        self.model_name = None
        self.GAN = None
        self.outputRoot = Path(__file__) / default_output_dir / "stylegan-images"
        self.outputRoot.mkdir(parents=True, exist_ok=True)

    ## methods called by UI
    def generate_image_from_ui(self, model_name: str, seed: int,
                                     psi: float) -> (Image.Image, str):
        self.set_model(model_name)

        if seed == -1:
            seed = self.newSeed()
        seedTxt = 'Seed: ' + str(seed)

        img = self.generate_image(seed, psi, global_state.image_pad)

        return img, seedTxt
        

    def generate_mix_from_ui(self, model_name: str, seed1: int, seed2: int,
                                     psi: float, interpType: str, mix: float) -> np.ndarray:
        self.set_model(model_name)

        if seed1 == -1:
            seed1 = self.newSeed()
        if seed2 == -1:
            seed2 = self.newSeed()

        img1, img2, img3 = self.generate_image_mix(seed1, seed2, psi, interpType, mix, global_state.image_pad)
        seedTxt1 = f"Seed 1: {str(seed1)} ({str_utils.num2hex(seed1)})"
        seedTxt2 = f"Seed 2: {str(seed2)} ({str_utils.num2hex(seed2)})"
 
        return img1, img2, img3, seedTxt1, seedTxt2

    def set_model(self, model_name: str) -> None:
        self.device = global_state.device

        if model_name != self.model_name:
            self.model_name = model_name
            self.output_path().mkdir(parents=True, exist_ok=True)
            path = file_utils.model_path / model_name
            self.GAN = GanModel(path, self.device)
            logger(f"Loaded model {model_name}")


    ## Image generation methods

    def generate_image(self, seed: int, psi: float, pad: float=1.0) -> Image.Image:
        params = {'seed': seed, 'psi': psi}
        output = self.find_or_generate_base_image(**params)
        if pad != 1.0:
            output = self.pad_image(output, pad)
            padded_path = f"base-{seed}-{psi}-pad{pad}.{global_state.image_format}"
            self.save_image_to_file(output, padded_path, params)
            # logger(f"  Padded image by {pad}x")
        return output

    def generate_image_mix(self, seed1: int, seed2: int, psi: float, interpType: str, mix: float, pad: float) -> np.ndarray:
        params = {'seed1': seed1, 'seed2': seed2, 'mix': mix, 'interp': interpType}

        img1, w1 = self.find_or_generate_base_image_and_weights(seed1, psi)
        img2, w2 = self.find_or_generate_base_image_and_weights(seed2, psi)

        basename = f"mix-{seed1}-{seed2}-mix{mix}-{interpType}"
        filename = f"{basename}.{global_state.image_format}"
        img3 = self.find_output_image(filename)
        if img3 is None:
            w_mix = self.mix_weights(w1, w2, mix, interpType)
            img3 = self.GAN.w_to_image(w_mix)
            self.save_image_to_file(img3, filename, params)

        if pad == 1.0:
            return img1, img2, img3

        img3p = self.pad_image(img3,pad)
        filename = f"{basename}-pad{pad}.{global_state.image_format}"
        self.save_image_to_file(img3p, filename, params)
        
        return img1, img2, img3p
        
    def pad_image(self, image: Image.Image, factor: float=1.0) -> Image.Image:
        resolution = self.GAN.img_resolution
        new_size = int(resolution*factor)
        padImage = Image.new(image.mode, (new_size, new_size), (0, 0, 0))
        padding = int((new_size-resolution)/2)
        padImage.paste(image, box=(padding, padding))
        return padImage

    def base_image_path(self, seed: int, psi: float) -> str:
        return f"base-{seed}-{psi}.{global_state.image_format}"

    def output_path(self):
        return self.outputRoot / ".".join(self.model_name.split(".")[:-1])

    def find_or_generate_base_image(self, seed: int, psi: float) -> Image.Image:
        params = {'seed': seed, 'psi': psi}
        msg = f"Rendered with {str(params)}"

        output = self.find_output_image(self.base_image_path(**params))
        if output is None:
            output, _ = self.generate_base_image(**params)
        else:
            msg += " (cached on disk)"

        logger(msg)

        return output

    def find_or_generate_base_image_and_weights(self, seed: int, psi: float) -> (Image.Image, torch.Tensor):
        params = {'seed': seed, 'psi': psi}
        msg = f"Rendered with {str(params)}"

        output = self.find_output_image(self.base_image_path(**params))
        if output is None:
            output, w = self.generate_base_image(**params)
        else:
            # load metadata from image file
            p = metadata.parse_params_from_image(output)
            w = p.get('tensor')
            if w is not None:
                w = str_utils.str2tensor(w).to(self.device)
                logger(f"Tensor found in metadata: {w.shape}")
            else:
                logger("Tensor not found... regenerating")
                _ , w = self.generate_base_image(**params)
            msg += " (cached on disk)"

        logger(msg)

        return output, w

    def find_output_image(self, filename: str) -> Union[None, Image.Image]:
        path = self.output_path() / filename
        if path.exists():
            return Image.open(path)

    # Make note that there are two return values here!
    def generate_base_image(self, seed: int, psi: float) -> (Image.Image, torch.Tensor):
        params = {'seed': seed, 'psi': psi}
        w = self.GAN.get_w_from_seed(**params)
        img = self.GAN.w_to_image(w)
        path = self.base_image_path(**params)
        params['tensor'] = str_utils.tensor2str(w)
        self.save_image_to_file(img, path, params)

        return img, w

    def save_image_to_file(self, image: Image.Image, filename: str, params: dict = None):
        path = self.output_path() / filename
        info = {
            'model': self.model_name,
            **params,
            'extension': 'gan-generator',
        }
        save_image_with_geninfo(image, str(info), str(path))

    ### Class Methods

    @classmethod
    def newSeed(cls) -> int:
        return random.randint(0, 0xFFFFFFFF - 1)

    @classmethod
    def xfade(cls, a,b,x):
        return a*(1.0-x) + b*x # basic linear interpolation

    @classmethod
    def mix_weights(cls, w1: torch.Tensor, w2: torch.Tensor, amt: float, mask: Union[str,int]) -> torch.Tensor:
        if isinstance(mask, str):
            match mask:
                case "coarse":
                    mask = 0xFF00
                case "mid":
                    mask = 0x0FF0
                case "fine":
                    mask = 0x00FF
                case "total":
                    mask = 0xFFFF
                case _:
                    mask = str_utils.str2num(mask)

        w_mix = w1.clone() # transfer onto L image as default

        if mask != 0xFFFF:
            if amt > 0: # transfer L onto R
                w_mix = w2.clone()
            else: # transfer R onto L
                i = abs(amt)
                w1,w2 = w2,w1 # swap L and R

        mask = cls.num2mask(mask)
        w_mix[:,mask,:] = cls.xfade(w1[:,mask,:], w2[:,mask,:], amt)

        return w_mix

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
