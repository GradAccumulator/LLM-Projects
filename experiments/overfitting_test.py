from models import Transformer
import torch, torch.nn as nn
import hydra
from omegaconf import DictConfig
from utils import dev_utils, nn_utils

@hydra.main(version_base=None, config_path="../configs", config_name="test_overfitting")
def main(cfg:DictConfig):
    device, dtype = cfg.train.model_device, nn_utils.load_dtype(cfg.train.precision)

    numel = sum(p.numel() for p in Transformer(cfg).parameters())
    print(f"Model parameter: {numel/1e6:.2f}M")

    model = Transformer(cfg).to(device, dtype)
    x = torch.randint(0, cfg.model.vocab_size, (10, cfg.train.batch_size, cfg.model.max_seq_len,), device=device)
    criterion = nn.CrossEntropyLoss()
    optim = nn_utils.build_optimizer(model.parameters(), cfg.optimizer.name, cfg.optimizer)
    scheduler = nn_utils.build_scheduler(optim, cfg=cfg)

    max_epochs = cfg.train.max_steps
    log_interval = cfg.train.log_interval
    for i in range(max_epochs):
        for k in range(len(x)):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x[k])
                loss = criterion(logits.view(-1, cfg.model.vocab_size), x[k].view(-1))
            loss.backward()
            optim.step()
            optim.zero_grad()
        scheduler.step()
        if (i-1)%log_interval == 0:
            print(
                f"{i+1}/{max_epochs}, loss= {loss.item()}"
            )




if __name__ == "__main__":
    main()