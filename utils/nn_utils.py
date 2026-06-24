import torch
from torch import Tensor
from omegaconf import DictConfig

def init_tensor(*shape:int, init_cfg:DictConfig|dict):
    match init_cfg.method:
        case "zeros":
            return torch.zeros(*shape)
        case "normal":
            return torch.normal(0, init_cfg['std'])
        case _:
            return torch.randn(*shape)