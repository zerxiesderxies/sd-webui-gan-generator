from typing import Union
import re
import io
import base64
import zlib
import hashlib
import torch
import numpy as np
from .global_state import logger

def str2num(string) -> Union[int, None]:
    # find a number at the end of a string, optionally enclosed in parentheses
    num_match = re.search(r'(?i)(-?0x[0-9a-f]+|-?\d+)\)?$', string)
    if num_match:
        num_str = num_match.group().strip("()")
        if num_str.startswith("0x") or num_str.startswith("-0x"):
            return int(num_str, 16)  # Convert hexadecimal string to integer
        else:
            return int(num_str)  # Convert decimal string to integer
    else:
        return None

def num2hex(num: int ) -> str:
    return str(hex(num)) #.upper().replace('0X', '0x')

def num2base(num: int, base: int=36) -> str:
    return np.base_repr(number, base)

def tensor2str(tensor: Union[torch.Tensor, np.ndarray]) -> str:
    if isinstance(tensor, torch.Tensor):
        # logger("converting to numpy")
        tensor = tensor.cpu().numpy()
    with io.BytesIO() as f:
        np.save(f, tensor)
        tensor_bytes = f.getvalue()
    compressed_bytes = zlib.compress(tensor_bytes)
    encoded_bytes = base64.b64encode(compressed_bytes)

    return encoded_bytes.decode('utf-8')

def str2tensor(encoded: str) -> torch.Tensor:
    # if not str.startswith("eJzt1"):
    #     logger("Vector is malformed: ", encoded[:5], "... ignoring")
    #     return torch.Tensor(0)

    decoded_bytes = base64.b64decode(encoded.strip())
    decompressed_bytes = zlib.decompress(decoded_bytes)
    with io.BytesIO(decompressed_bytes) as f:
        tensor = np.load(f)

    return torch.tensor(tensor)

def crc_hash(string: str) -> str:  # 8 characters
    crc = zlib.crc32(string.encode())
    return format(crc & 0xFFFFFFFF, '08x')

def sha_hash(string: str) -> str:    # 64 characters
    return hashlib.sha256(string.encode()).hexdigest()
