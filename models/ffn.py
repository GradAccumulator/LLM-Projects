import torch, torch.nn as nn
from torch     import Tensor
from typing    import Literal
from omegaconf import DictConfig

from .mul      import mul
from .silu     import SiLU
from .linear   import Linear
from utils     import dev_utils

class FFN(nn.Module):
    def __init__(
        self, 
        in_features :int, 
        ffn_dim     :int,
        ffn_type    :Literal['swiglu', 'mlp']='swiglu',
        activation  :str                     =None, 
        init_cfg    :DictConfig|dict         =None, 
        use_bias    :bool                    =False
    ):
        '''```
        init_cfg = {
            'linear1': {
                "weight":{
                    "method":...
                },
                "bias":{
                    "method":...
                }
            },
            'linear2': {
                "weight":{
                    "method":...
                },
                "bias":{
                    "method":...
                }
            }
        }
        ```'''
        super().__init__()
        
        dev_utils.type_check(
            ("in_features"  , in_features   , int),
            ("ffn_dim"      , ffn_dim       , int),
            ("use_bias"     , use_bias      , bool),
            ("activation"   , activation    , str|None),
            ("init_cfg"     , init_cfg      , DictConfig|dict|None),
            func_name="FFN.__init__()"
        )
        if ffn_type not in ["swiglu", "mlp"]:
            raise ValueError(f"<FFN.__init__()> ffn_type이 지원되지 않는 유형입니다. 현재: {ffn_type}")
        
        self._use_bias = use_bias
        
        init_cfg = dev_utils.make_dictconfig(init_cfg, default={
            'linear1':None,
            'linear2':None
        })
        dev_utils.check_dictconfig(
            init_cfg,
            ("linear1", "linear2"),
            "FFN.__init__()"
        )
        
        match activation:
            case 'silu'|None:
                self.act_fn = SiLU()
            case _:
                raise ValueError(f"<FFN.__init__()> activation 인자가 지원되지 않는 유형입니다 : {activation}")
        
        self._ffn_type   = ffn_type
        self._use_swiglu = ffn_type == 'swiglu'
        if self.use_swiglu:
            self.linear1 = Linear(
                in_features, 
                ffn_dim*2, 
                init_cfg    =init_cfg.linear1,
                use_bias    =use_bias
            )
        else:
            self.linear1 = Linear(
                in_features, 
                ffn_dim, 
                init_cfg    =init_cfg.linear1,
                use_bias    =use_bias
            )
        self.linear2 = Linear(
            ffn_dim, 
            in_features, 
            init_cfg    =init_cfg.linear2,
            use_bias    =use_bias
        )
    
    def forward(self, x:Tensor)->Tensor:
        #x.shape == (B, T, D)
        if self.use_swiglu:
            l1,l2 = self.linear1(x).chunk(2, dim=-1)
            l1    = self.act_fn(l1)
            
            x = self.linear2(mul(l1, l2))
        
        else:
            x = self.linear1(x)
            
            x = self.act_fn(x)
            
            x = self.linear2(x)
        
        return x
    
    @property
    def ffn_dim(self): return self.linear2.in_features
    @property
    def use_bias(self): return self._use_bias
    @property
    def ffn_type(self): return self._ffn_type
    @property
    def use_swiglu(self): return self._use_swiglu
    @property
    def in_features(self): return self.linear1.in_features
    @property
    def out_features(self): return self.linear2.out_features