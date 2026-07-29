from models import Transformer
import torch, torch.nn as nn
import hydra
from omegaconf import DictConfig
from utils import dev_utils, nn_utils


def print_header(title: str, width: int = 60):
    print(f"\n{f' {title} ':=^{width}}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_dataset(cfg: DictConfig, device: str) -> torch.Tensor:
    return torch.randint(
        low=0,
        high=cfg.model.vocab_size,
        size=(
            cfg.train.num_batches,
            cfg.train.batch_size,
            cfg.model.max_seq_len,
        ),
        device=device,
        dtype=torch.long,
    )


def compute_loss(
    model: Transformer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    vocab_size: int,
) -> torch.Tensor:

    logits = model(inputs)

    return criterion(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
    )


def train_one_epoch(
    model: Transformer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
) -> float:
    model.train()
    total_loss = 0.0

    for i, batch in enumerate(inputs):
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            loss = compute_loss(
                model=model,
                inputs=batch,
                targets=targets[i],
                criterion=criterion,
                vocab_size=cfg.model.vocab_size,
            )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(inputs)


@hydra.main(version_base=None, config_path="../configs", config_name="test_overfitting")
def main(cfg: DictConfig):
    print_header("OVERFIT TEST")

    device, dtype = cfg.train.model_device, nn_utils.load_dtype(cfg.train.precision)

    model = Transformer(cfg).to(device=device, dtype=dtype)
    numel = count_parameters(model)
    numel = f"{numel/1e9:.2f}B" if numel > 1e9 else f"{numel/1e6:.2f}M"
    print(f"Model parameters: {numel}")

    data = build_dataset(cfg, device)
    inputs = data[..., :-1]
    targets = data[..., 1:]

    criterion = nn.CrossEntropyLoss()
    optimizer = nn_utils.build_optimizer(
        model.parameters(),
        cfg.optimizer.name,
        cfg=cfg.optimizer,
    )
    scheduler = nn_utils.build_scheduler(optimizer, cfg=cfg)

    max_steps = cfg.train.max_steps
    log_interval = cfg.train.log_interval

    for epoch in range(1, max_steps + 1):
        avg_loss = train_one_epoch(
            model=model,
            inputs=inputs,
            targets=targets,
            criterion=criterion,
            optimizer=optimizer,
            cfg=cfg,
        )

        scheduler.step()

        if epoch % log_interval == 0 or epoch == 1:
            loss_str = f"{avg_loss:.4e}" if avg_loss < 1e-3 else f"{avg_loss:.4f}"
            print(
                f"epoch {epoch:>{len(str(max_steps))}d}/{max_steps} | loss= {loss_str}"
            )


if __name__ == "__main__":
    main()
