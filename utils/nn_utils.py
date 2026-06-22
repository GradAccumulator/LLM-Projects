import torch
from torch import Tensor
from omegaconf import DictConfig

def init_tensor(*shape, init_cfg:DictConfig):
    match init_cfg.method:
        case "zeros":
            return torch.zeros(*shape)
        case "normal":
            return torch.randn(*shape)*init_cfg.std
        case _:
            return torch.randn(*shape)