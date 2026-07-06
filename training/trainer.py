import torch, torch.nn as nn
from torch import Tensor
from models import Transformer
import torch.utils.data as data

class TransformerTrainer:
    def __init__(
            self,
            model:Transformer, 
            trainset:data.DataLoader, 
            valset:data.DataLoader,
        ):
        pass