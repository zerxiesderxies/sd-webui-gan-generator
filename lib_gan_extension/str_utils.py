from typing import Union
import re
import numpy as np

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
