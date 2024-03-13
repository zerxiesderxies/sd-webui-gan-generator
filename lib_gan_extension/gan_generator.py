from __future__ import annotations
from typing import Union
from pathlib import Path
import numpy as np
from PIL import Image
import random

from modules.images import save_image_with_geninfo
from modules.paths_internal import default_output_dir

from lib_gan_extension import GanModel, global_state, file_utils, str_utils
from .global_state import logger

class GanGenerator:
    def __init__(self):
        self.device = None
        self.model_name = None
        self.GAN = None
        self.outputRoot = Path(__file__) / default_output_dir / "stylegan-images"
        file_utils.mkdir_p(self.outputRoot)

    ## methods called by UI
    def generate_image_from_ui(self, device: str, model_name: str, seed: int,
                                     psi: float, pad: float) -> np.ndarray:        
        self.prepare_model(model_name, device)

        if seed == -1:
            seed = self.newSeed()
        seedTxt = 'Seed: ' + str(seed)

        return self.generate_image(seed, psi, pad), seedTxt
        

    def generate_mix_from_ui(self, device: str, model_name: str, seed1: int, seed2: int,
                                     psi: float, interpType: str, mix: float, pad: float) -> np.ndarray:
        self.prepare_model(model_name, device)

        if seed1 == -1:
            seed1 = self.newSeed()
        if seed2 == -1:
            seed2 = self.newSeed()

        img1, img2, img3 =  self.generate_image_mix(seed1, seed2, psi, interpType, mix, pad)
        seedTxt1 = f"Seed 1: {str(seed1)} ({str_utils.num2hex(seed1)})"
        seedTxt2 = f"Seed 2: {str(seed2)} ({str_utils.num2hex(seed2)})"
 
        return img1, img2, img3, seedTxt1, seedTxt2


    def prepare_model(self, model_name: str, device: str) -> None:
        if device != self.device:
            self.device = device
            logger(f"Device selected: {device}")
        if model_name != self.model_name:
            self.model_name = model_name
            file_utils.mkdir_p(self.output_path())
            path = file_utils.model_path / model_name
            self.GAN = GanModel(str(path), self.device)
            logger(f"Loaded model {model_name}")
        else:
            self.GAN.set_device(self.device)


    ## Image generation methods

    def generate_image(self, seed: int, psi: float, pad: float=1.0) -> Image.Image:
        params = {'seed': seed, 'psi': psi}
        output, _ = self.find_or_generate_base_image(**params)
        if pad != 1.0:
            output = self.pad_image(output, pad)
            padded_path = f"base-{seed}-{psi}-pad{pad}.{global_state.image_format}"
            self.save_image_to_file(output, padded_path, {**params, 'pad': pad})
            # logger(f"  Padded image by {pad}x")
        return output

    def generate_image_mix(self, seed1: int, seed2: int, psi: float, interpType: str, mix: float, pad: float) -> np.ndarray:
        img1, w1 = self.find_or_generate_base_image(seed1, psi)
        img2, w2 = self.find_or_generate_base_image(seed2, psi)

        slider_max = 2.0 # FIXME: this is a hack to fix the slider bug where range is stuck at 0-2
        w_mix = self.mix_weights(w1, w2, mix/slider_max, interpType)
        img3 = self.GAN.w_to_image(w_mix)

        param_str = f"mix-{seed1}-{seed2}-{mix}-{interpType}"
        filename = f"#{param_str}.{global_state.image_format}"
        self.save_image_to_file(img3, filename, params={'seed1': seed1, 'seed2': seed2, 'mix': mix, 'interp': interpType})

        img3p = self.pad_image(img3,pad)
        filename = f"#{param_str}-pad{pad}.{global_state.image_format}"
        self.save_image_to_file(img3p, filename, params={'seed1': seed1, 'seed2': seed2, 'mix': mix, 'interp': interpType, 'pad': pad})
        
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

        output = self.find_image_if_exists(self.base_image_path(**params))
        if output is None:
            output, w = self.generate_base_image(**params)
        else:
            msg += " (cached on disk)"
        logger(msg)

        return output, w

    def find_image_if_exists(self, filename: str) -> Union[None, Image.Image]:
        path = self.output_path() / filename
        if path.exists():
            return Image.open(path)
        return None

    def generate_base_image(self, seed: int, psi: float) -> (Image.Image, torch.Tensor):
        params = {'seed': seed, 'psi': psi}
        w = self.GAN.get_w_from_seed(**params)
        img = self.GAN.w_to_image(w)
        self.save_image_to_file(img, self.base_image_path(**params), params)
        return img, w

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
            amt = amt * 2.0 - 1.0 # rescale between -1 and 1
            if amt > 0: # transfer L onto R
                w_mix = w2.clone()
            else: # transfer R onto L
                i = abs(amt)
                w1,w2 = w2,w1 # swap L and R
            amt *= 1.5 # increase range

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

