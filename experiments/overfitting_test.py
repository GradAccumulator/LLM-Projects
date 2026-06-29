from models import Transformer
import torch, torch.nn as nn
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../configs", config_name="test_overfitting")
def main(cfg:DictConfig):
    model = Transformer(cfg)
    print(model)
    x = torch.randint(0, cfg.model.vocab_size, (2, cfg.model.max_seq_len))
    out = model(x)
    print(out.shape)

if __name__ == "__main__":
    main()