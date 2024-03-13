import ast
from PIL import Image
from .global_state import logger

def parse_params_from_image(img: [str,Image.Image]) -> dict:
    if isinstance(img, str):
        img = Image.open(img)
    p = img.info
    if "gan-generator" in str(p):
        # some weird stuff here for for legacy images with bad metadata
        if isinstance( p.get('parameters'), str ):
            p['parameters'] = ast.literal_eval(p.get('parameters'))
        p = peel_parameters( p )
        # logger(f"loading image params: {repr(p)}")
        return p

def peel_parameters(data): # recursively peel 'parameters' from nested dict
    if isinstance(data, dict):
        if 'parameters' in data:
            return peel_parameters(data['parameters'])
        return {k: peel_parameters(v) for k, v in data.items()}
    return data
