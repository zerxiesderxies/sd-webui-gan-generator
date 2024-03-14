from __future__ import annotations
from typing import Union
import ast
from PIL import Image
from .global_state import logger

import modules
from modules.images import read_info_from_image

def parse_params_from_image(img: Union[str,Image.Image]) -> dict:
    if isinstance(img, str):
        img = Image.open(img)
    geninfo,_ = read_info_from_image(img)
    if geninfo is not None:
        p = ast.literal_eval(geninfo)
        while 'parameters' in p:
            p = p.pop('parameters',None)
        # logger(f"loaded {img.format} params: {repr(p)}")
        if p['extension'] == 'gan-generator':
            return p
    else:
        return {}
