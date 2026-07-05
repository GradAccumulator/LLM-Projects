import torch
from typing     import Any
from torch      import Tensor
from omegaconf  import OmegaConf, DictConfig

def type_check(*values:tuple[str, Any, type], func_name, strict=True):
    for value in values:
        if not isinstance(value[1], value[2]):
            if strict:
                raise TypeError(f"<{func_name}> {value[0]}의 타입이 부적절합니다. 예상한 타입:{value[2]}, 현재: {type(value[1])}")
            return False
    return True

def make_dictconfig(arg:DictConfig|dict|None, default:dict|None=None)->DictConfig:
    type_check(
        ("arg"      , arg       , DictConfig|dict|None), 
        ("default"  , default   , dict|None), 
        func_name="make_dictconfig()"
    )
    
    if isinstance(arg, DictConfig):
        return arg
    elif arg is None:
        return OmegaConf.create(default)
    else:
        return OmegaConf.create(arg)

def check_dictconfig(cfg:DictConfig|dict, required_keys:list, func_name:str):
    missing = [key for key in required_keys if OmegaConf.select(cfg, key) is None]
    if missing:
        raise ValueError(f"<{func_name}> cfg에서 누락된 key: {missing}")