import os
import torch
import torch.nn as nn
from torch import Tensor
import torch.utils.data as data
import hydra
from pathlib import Path

from utils import dev_utils, nn_utils
from tokenizer import Tokenizer
from models import Transformer
from .transformer_trainer import TransformerTrainer
from processing_datasets import LLMDataset

def build_dataloaders(cfg):
    dataset_dir = Path(__file__).resolve().parent.parent.parent/"datasets"
    train_dataset = LLMDataset(
        cfg.model.max_seq_len,
        datasets_dir =dataset_dir,
        dataset_name =cfg.dataset.name,
        total_tokens =cfg.dataset.total_tokens,
        dataset_type ="train",
        bin_dtype    =nn_utils.load_dtype(cfg.dataset.bin_dtype)
    )
    val_dataset = LLMDataset(
        cfg.model.max_seq_len,
        datasets_dir =dataset_dir,
        dataset_name =cfg.dataset.name,
        total_tokens =cfg.dataset.validation_tokens,
        dataset_type ="test",
        bin_dtype    =nn_utils.load_dtype(cfg.dataset.bin_dtype)
    )

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=cfg.validation.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    return train_loader, val_loader

model_params = 1#B
vocab_size = 32#K
seq_len = 1024
model_num = 1

model_name = f"model{model_params}_v{vocab_size}_s{seq_len}-{model_num}"
@hydra.main(version_base=None, config_path="../../configs", config_name=model_name)
def main(cfg):
    nn_utils.resolve_llm_cfg(cfg)
    model = Transformer(cfg).to(
        cfg.train.model_device, 
        nn_utils.load_dtype(cfg.train.model_dtype),
    )
    optimizer = nn_utils.build_optimizer(model.parameters(), cfg.optimizer.name, cfg.optimizer)
    scheduler = nn_utils.build_scheduler(optimizer, cfg)
    loss_fn = nn_utils.build_loss_fn(cfg.loss.name, cfg.loss)
    tokenizer = Tokenizer(f"{vocab_size}k_"+cfg.dataset.name)
    dataloaders = build_dataloaders(cfg)

    trainer = TransformerTrainer(
        model,
        dataloaders,
        optimizer,
        scheduler,
        loss_fn,
        cfg,
        tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main()