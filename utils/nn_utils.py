import torch
from torch import Tensor
from omegaconf import DictConfig
import omegaconf
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
from . import dev_utils
from typing import overload
import math

def init_tensor(*shape:int, init_cfg:DictConfig|dict):
    match init_cfg.method:
        case "zeros":
            return torch.zeros(*shape)
        case "ones":
            return torch.ones(*shape)
        case "normal":
            return torch.randn(*shape) * init_cfg['std']
        case method:
            raise ValueError(
                f"<init_tensor()> 전달된 초기화 방식이 잘못되었습니다. : {method}"
            )

@overload
def build_optimizer(params, name:str, cfg:DictConfig=None) -> optim.Optimizer:
    '''```
    "cfg" = {
        "name":N,
        N: {
            "lr": ..., 
            "weight_decay":..., 
            ...
        }
    }
    ```'''
    ...
@overload
def build_optimizer(params, name:str, *args, **kwargs) -> optim.Optimizer:...
def build_optimizer(params, name:str, *args, cfg:DictConfig=None, **kwargs) -> optim.Optimizer:
    dev_utils.type_check(
        ("name", name, str),
        func_name="build_optimizer()"
    )

    optimizer_cls = getattr(optim, name, None)

    if optimizer_cls is None:
        raise ValueError(
            f"<build_optimizer()> torch.optim에 name={name}인 optimizer가 없습니다."
        )

    if not isinstance(optimizer_cls, type) or not issubclass(optimizer_cls, optim.Optimizer):
        raise ValueError(
            f"<build_optimizer()> torch.optim.{name}은 Optimizer 클래스가 아닙니다."
        )
    
    has_positional_cfg = len(args) > 0 and isinstance(args[0], DictConfig|dict)
    if has_positional_cfg or cfg is not None :
        if has_positional_cfg:
            cfg = args[0]
            args = args[1:]
        else:
            dev_utils.type_check(
                ("cfg", cfg, DictConfig|dict),
                func_name="build_optimizer()"
            )
        cfg = dev_utils.make_dictconfig(cfg)
        if cfg.get(f"{name}") is None:
            raise ValueError(
                f"<build_optimizer()> cfg.optimizer에 {name} 설정이 없습니다."
            )
        optimizer_kwargs = dict(cfg[name])
        optimizer_kwargs.update(kwargs)
        return optimizer_cls(params=params, *args, **optimizer_kwargs)
    if args == () and kwargs == {}:
        raise ValueError(
            f"<build_optimizer()> cfg.optimizer에 {name} 설정이 없습니다."
        )
    return optimizer_cls(params=params, *args, **kwargs)

@overload
def build_scheduler(optimizer:optim.Optimizer, cfg:DictConfig) -> optim.lr_scheduler.LRScheduler:
    '''```
    cfg = {
        "train": {
            "max_steps":...
        },
        "scheduler": {
            "warmup_ratio":...,
            "min_lr_ratio":...
        }
    }
    ```'''
    ...
@overload
def build_scheduler(optimizer:optim.Optimizer, max_steps:int, warmup_ratio:float|int, min_lr_ratio:float|int) -> optim.lr_scheduler.LRScheduler:...
def build_scheduler(optimizer:optim.Optimizer, max_steps:int=None, warmup_ratio:float|int=None, min_lr_ratio:float|int=None,*,cfg:DictConfig=None) -> optim.lr_scheduler.LRScheduler:
    if cfg is not None or isinstance(max_steps, DictConfig|dict):
        if cfg is not None:
            dev_utils.type_check(
                ("cfg", cfg, DictConfig|dict),
                func_name="build_scheduler()"
            )
        else:
            cfg = max_steps

        cfg = dev_utils.make_dictconfig(cfg)
        dev_utils.check_dictconfig(cfg, ('train.max_steps', 'scheduler.warmup_ratio', 'scheduler.min_lr_ratio'), "build_scheduler()")
        max_steps    = cfg.train.max_steps
        warmup_ratio = cfg.scheduler.warmup_ratio
        min_lr_ratio = cfg.scheduler.min_lr_ratio
    else:
        none_list = []
        if max_steps is None:
            none_list.append("max_steps")
        if warmup_ratio is None:
            none_list.append('warmup_ratio')
        if min_lr_ratio is None:
            none_list.append('min_lr_ratio')

        if none_list:
            raise ValueError(
                "cfg를 인자로 전달하지 않았을 때에는 max_steps, warmup_ratio, min_lr_ratio 인자를 전달해야 합니다."
                f"\n현재 전달되지 않은 인자들: {none_list}"
            )

    dev_utils.type_check(
        ("optimizer"    , optimizer     , optim.Optimizer),
        ("max_steps"    , max_steps     , int),
        ("warmup_ratio" , warmup_ratio  , float|int),
        ("min_lr_ratio" , min_lr_ratio  , float|int),
        func_name="build_scheduler()"
    )

    if max_steps <= 0:
        raise ValueError("<build_scheduler()> max_steps는 양수여야 합니다.")

    if not (0 <= warmup_ratio < 1):
        raise ValueError("<build_scheduler()> warmup_ratio는 [0, 1)의 실수여야 합니다.")

    if not (0 <= min_lr_ratio <= 1):
        raise ValueError("<build_scheduler()> min_lr_ratio는 [0, 1]의 실수여야 합니다.")

    warmup_steps = int(max_steps * warmup_ratio)
    decay_steps  = max(max_steps - warmup_steps, 1)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(step / warmup_steps, 1e-3)

        progress = (step - warmup_steps) / decay_steps
        progress = min(progress, 1.0)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda
    )

def load_dtype(name:str)->torch.dtype:
    dtype_map = {
        'bf16':torch.bfloat16,
        'bfloat16':torch.bfloat16,

        'fp16':torch.float16,
        'float16':torch.float16,

        'fp32':torch.float32,
        'float32':torch.float32,
        
        'half':torch.half,
        'float':torch.float,
        'double':torch.double
    }
    if name in dtype_map:
        return dtype_map[name]
    dtype = getattr(torch, name, default=None)
    if dtype is None:
        raise ValueError(
            f"<load_dtype()> {name}에 해당하는 torch의 dtype이 존재하지 않습니다."
        )
    if not isinstance(dtype, torch.dtype):
        raise ValueError(
            f"<load_dtype()> {name}은 torch의 dtype 인스턴스가 아닙니다."
        )
    
    return dtype