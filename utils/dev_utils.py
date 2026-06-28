import torch
from torch import Tensor
from typing import Any
from omegaconf import OmegaConf, DictConfig

def type_check(*values:tuple[str, Any, type], func_name, strict=True):
    for value in values:
        if not isinstance(value[1], value[2]):
            if strict:
                raise TypeError(f"<{func_name}> {value[0]}의 타입이 부적절합니다. 예상한 타입:{value[2]}, 현재: {type(value[1])}")
            return False
    return True

def make_dictconfig(arg:DictConfig|dict|None, default:dict|None=None)->DictConfig:
    type_check(
        ("arg", arg, DictConfig|dict|None), 
        ("default", default, dict|None), 
        func_name="make_dictconfig()"
    )
    
    if isinstance(arg, DictConfig):
        return arg
    elif arg is None:
        return OmegaConf.create(default)
    else:
        return OmegaConf.create(arg)

def check_dictconfig(arg:DictConfig|dict, needed_keys:list, func_name:str):
    def _get_all_keys(arg:DictConfig, keys:set=None):
        if keys is None:
            keys = set(arg.keys())
        else:
            keys.update(arg.keys())
        
        for v in arg.values():
            if isinstance(v, DictConfig):
                keys = _get_all_keys(v, keys=keys)
        
        return keys
        
    arg = make_dictconfig(arg)
    all_keys = _get_all_keys(arg)
    needed_keys = set(needed_keys if needed_keys is not None else [])
    
    for k in needed_keys:
        if k not in all_keys:
            raise KeyError(
                f"<{func_name}>"
                f" cfg의 key가 잘못 전달되었습니다."
                f" 예상한 key: {needed_keys}, 실제 key:{all_keys}"
            )