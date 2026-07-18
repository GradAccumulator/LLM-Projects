import math
import torch
import omegaconf
import torch.optim as optim
from typing import overload,Literal

from .          import dev_utils
from torch      import Tensor
from omegaconf  import DictConfig
from configs    import runtime as rt


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


def _quantize(x:Tensor, dtype, method) -> tuple[Literal[False],Tensor] | tuple[Literal[True],Tensor,Tensor]:
    if "float8" in str(dtype):
        match method:
            case 'per-tensor':
                scale = (x.abs().amax()/448.0).clamp_min(1e-12)
                quantized = (x/scale).to(dtype=dtype)
                return True,quantized,scale

    return False,x.to(dtype=dtype)

def _save_for_backward_debug(dtype, method, allowed_quantize_methods):
    if not isinstance(dtype, torch.dtype):
        raise ValueError(
            "<save_for_backward()._quantize()._quantize_debug()> configs/runtime.py의 ACTIVATION_SAVE_DTYPE이 torch.dtype이 아닙니다."
            f" 현재: {type(dtype)}"
        )

    if method not in allowed_quantize_methods:
            raise ValueError(
                "<save_for_backward()._quantize()._quantize_debug()> configs/runtime.py의 ACTIVATION_QUANTIZE_METHOD 설정이 잘못되었습니다."
                f" 현재: {rt.ACTIVATION_QUANTIZE_METHOD}, 허용되는 값: {allowed_quantize_methods}"
            )
def save_for_backward(ctx, *args):
    dtype = rt.ACTIVATION_SAVE_DTYPE
    method = rt.ACTIVATION_QUANTIZE_METHOD
    allowed_quantize_methods= [
        'per-tensor',
    ]

    if rt.DEBUG_CHECKS == True:
        _save_for_backward_debug(dtype, method, allowed_quantize_methods)

    quantized_tensors = []
    scales = []
    for arg in args:
        quantized = _quantize(arg, dtype=dtype, method=method)
        quantized_tensors.append(quantized[1])
        if quantized[0]:
            scales.append(quantized[2])
    ctx.tensors_length = len(quantized_tensors)
    ctx.save_for_backward(*quantized_tensors, *scales)

def dequantize(ctx) -> list[Tensor]:
    dtype = torch.get_autocast_gpu_dtype()
    quantized = ctx.saved_tensors
    quantized_tensors = quantized[:ctx.tensors_length]
    scales = quantized[ctx.tensors_length:]

    if scales:
        dequantized_tensors = []
        for tensor, scale in zip(quantized_tensors, scales):
            dequantized_tensors.append(tensor.to(dtype=dtype)*scale)
    else:
        dequantized_tensors = [tensor.to(dtype=dtype) for tensor in quantized_tensors]
    
    return dequantized_tensors

@overload
def build_loss_fn(name:str, cfg:DictConfig|dict):
    '''
    cfg = {
        "name": N,
        N : ...
    }
    '''
    ...
@overload
def build_loss_fn(name, *args, **kwargs):...
def build_loss_fn(name, *args, cfg:DictConfig|dict=None, **kwargs):
    func_name = "build_loss_fn()"

    dev_utils.type_check(
        ("name", name, str),
        func_name=func_name
    )
    loss_fn_cls = torch.nn.__dict__.get(name, None)
    if loss_fn_cls is not None:
        if not (
            isinstance(loss_fn_cls,type) 
            and issubclass(loss_fn_cls, torch.nn.Module)
            and 'loss' in name.lower()
        ):
            raise ValueError(
                f"<{func_name}> name={name}은 torch.nn의 손실함수 클래스가 아닙니다."
            )
    else:
        raise ValueError(
            f"<{func_name}> torch.nn에 name= {name}에 해당하는 객체/클래스/모듈이 존재하지 않습니다."
        )
    

    if cfg is not None:
        dev_utils.type_check(
            ("cfg", cfg, DictConfig|dict),
            func_name=func_name
        )
        _kwargs = kwargs
        kwargs = cfg.get(name, None)
        kwargs.update(_kwargs)
        
    return loss_fn_cls(*args, **kwargs)

def resolve_llm_cfg(cfg:DictConfig):
    if isinstance(cfg.dataset.total_tokens, str):
        cfg.dataset.total_tokens = total_tokens = dev_utils.str_to_num(cfg.dataset.total_tokens)
    if cfg.train.validation == None:
        cfg.train.validation = cfg.train.validation_interval>0
    if cfg.train.max_steps == 0 or cfg.train.max_steps is None:
        cfg.max_steps = total_tokens//(cfg.model.max_seq_len*cfg.train.batch_size)
    if total_tokens%(cfg.model.max_seq_len*cfg.train.batch_size) != 0:
        print(f"현재 total_tokens= {dev_utils.num_to_str(total_tokens)}가 seq_len*batch_size= {cfg.model.seq_len*cfg.train.batch_size}와 나누어 떨어지지 않습니다.")
        total_tokens = (total_tokens//(cfg.model.max_seq_len*cfg.train.batch_size))*(cfg.model.max_seq_len*cfg.train.batch_size)
        print(f"total_tokens를 {total_tokens}로 재설정합니다.")
        cfg.dataset.total_tokens = total_tokens