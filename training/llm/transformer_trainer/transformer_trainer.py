from torch import Tensor
from queue import Queue, Empty
from typing import overload
from omegaconf import DictConfig
import math
import torch, torch.nn as nn
import torch.utils.data as data

from . import commands
from models import Transformer
from .keyboard_listener import KeyboardListener
from utils import dev_utils, nn_utils
from .commands import TrainLoopResult, TrainingCommand
from tokenizer import Tokenizer
from utils.logging import LoggerList, TrainingEvent


class TransformerTrainer:
    @overload
    def __init__(
        self,
        model: Transformer,
        dataloaders: (
            tuple[data.DataLoader, data.DataLoader]
            | list[data.DataLoader, data.DataLoader]
        ),
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss_fn: nn.Module,
        cfg: DictConfig | dict,
        *,
        tokenizer: Tokenizer = None,
        loggers: LoggerList = None,
    ): ...
    @overload
    def __init__(
        self,
        model: Transformer,
        dataloaders: (
            tuple[data.DataLoader, data.DataLoader]
            | list[data.DataLoader, data.DataLoader]
        ),
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss_fn: nn.Module,
        log_interval: int,
        max_steps: int,
        tokenizer: Tokenizer = None,
        loggers: LoggerList = None,
        precision: str = "fp32",
        grad_clip_norm: float | int = 1.0,
        grad_accumulation: int = 1,
        validation_interval: int | None = None,
        validation: bool = True,
    ): ...
    def __init__(
        self,
        model: Transformer,
        dataloaders: (
            tuple[data.DataLoader, data.DataLoader]
            | list[data.DataLoader, data.DataLoader]
        ),
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss_fn: nn.Module,
        log_interval: int | DictConfig | dict,
        max_steps: int = None,
        tokenizer: Tokenizer = None,
        loggers: LoggerList = None,
        precision: str = "fp32",
        grad_clip_norm: float | int = 1.0,
        grad_accumulation: int = 1,
        validation_interval: int | None = None,
        validation: bool = True,
        cfg: DictConfig | dict = None,
    ):
        func_name = "TransformerTrainer.__init__()"
        dev_utils.type_check(
            ("loss_fn", loss_fn, nn.Module),
            ("loggers", loggers, LoggerList),
            ("dataloaders", dataloaders, tuple | list),
            ("model", model, Transformer),
            ("tokenizer", tokenizer, Tokenizer | None),
            ("dataloaders[0]", dataloaders[0], data.DataLoader),
            ("dataloaders[1]", dataloaders[1], data.DataLoader),
            ("optim", optimizer, torch.optim.Optimizer),
            ("scheduler", scheduler, torch.optim.lr_scheduler.LRScheduler),
            func_name=func_name,
        )
        self.model = model
        self.loss_fn = loss_fn
        self.optim = optimizer
        self.scheduler = scheduler
        self.train_loader = dataloaders[0]
        self.valset_loader = dataloaders[1]
        self.tokenizer = tokenizer
        self.loggers = loggers or LoggerList()

        if isinstance(cfg, DictConfig | dict) or isinstance(
            log_interval, DictConfig | dict
        ):
            if isinstance(cfg, DictConfig | dict):
                cfg = dev_utils.make_dictconfig(cfg)
            elif isinstance(log_interval, DictConfig | dict):
                cfg = dev_utils.make_dictconfig(log_interval)
            dev_utils.check_dictconfig(
                cfg,
                [
                    "train.max_steps",
                    "train.precision",
                    "train.validation",
                    "train.log_interval",
                    "train.validation_interval",
                    "optimizer.grad_clip_norm",
                    "optimizer.grad_accumulation",
                ],
                func_name="TransformerTrainer.__init__()",
            )
            max_steps = cfg.train.max_steps
            precision = cfg.train.precision
            validation = cfg.train.validation
            log_interval = cfg.train.log_interval
            validation_interval = cfg.train.validation_interval
            grad_clip_norm = cfg.optimizer.grad_clip_norm
            grad_accumulation = cfg.optimizer.grad_accumulation

        dev_utils.type_check(
            ("max_steps", max_steps, int),
            ("log_interval", log_interval, int),
            ("grad_accumulation", grad_accumulation, int),
            ("validation_interval", validation_interval, int),
            ("validation", validation, bool),
            ("grad_clip_norm", grad_clip_norm, float | int),
            ("precision", precision, str | torch.dtype),
            func_name=func_name,
        )

        self._max_steps = max_steps
        self.validation = validation
        self._log_interval = log_interval
        self._grad_clip_norm = grad_clip_norm
        self._precision = nn_utils.load_dtype(precision)
        self._grad_accumulation = grad_accumulation
        self.validation_interval = validation_interval

        self._tot_loss = 0.0
        self._tokens_seen = 0
        self._current_step = 0
        self._accumulated_batches = 0
        self._key_listener = KeyboardListener(
            key_map={
                "q": commands.CancelTrainingCommand,
                "b": commands.BreakpointCommand,
                "v": commands.ValidateCommand,
            },
            trainer=self,
        )
        self._command_queue: Queue[TrainingCommand] = Queue()

    def submit_command(self, command):
        self._command_queue.put(command)

    def cancel_training(self):
        return TrainLoopResult.USER_CANCELLED

    def _process_commands(self) -> TrainLoopResult | None:
        while True:
            try:
                command = self._command_queue.get_nowait()
            except Empty:
                break

            if (res := command.execute(self)) is not None:
                return res

    def _empty_cache(self):
        if self.model.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    @torch.no_grad()
    def validate(self):
        print("<TransformerTrainer.validate()> validation started")
        self.model.eval()
        self._empty_cache()
        tot_loss = 0.0
        total_dataset_length = 0

        for x, target in self.valset_loader:
            loss = self._forward(x, target)

            dataset_length = x.size(0) * x.size(1)
            tot_loss += loss * dataset_length
            total_dataset_length += dataset_length

            if (res := self._process_commands()) is not None:
                return res

        avg_loss = tot_loss.item() / total_dataset_length
        self._log(
            event=TrainingEvent.VALIDATION_ENDED,
            trainer=self,
            loss=avg_loss,
            ppl=math.exp(loss),
        )
        self._empty_cache()
        self.model.train()

    def _log(self, event, *args, **kwargs):
        self.loggers(*args, event=event, **kwargs)

    def stop_train(self):
        self._log(
            event=TrainingEvent.TRAIN_STOPPED,
            trainer=self,
        )

        breakpoint()

        self._log(
            event=TrainingEvent.TRAIN_RESUMED,
            trainer=self,
        )

    def _handle_step_conditions(self) -> TrainLoopResult | None:
        if self.current_step % self.log_interval == 0:
            self._log(
                event=TrainingEvent.LOG_INTERVAL_REACHED,
                trainer=self,
                loss=self._tot_loss / self._tokens_seen,
                tokens_seen=self.tokens_seen,
            )
            self._tokens_seen = 0
            self._tot_loss = 0.0

        if (
            self.validation
            and self.validation_interval is not None
            and self.current_step % self.validation_interval == 0
        ):
            if (res := self.validate()) is not None:
                return res

        if self.current_step >= self.max_steps:
            return TrainLoopResult.MAX_STEPS_REACHED

    def step(self):
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optim.step()
        self.scheduler.step()
        self.optim.zero_grad(set_to_none=True)

        self._current_step += 1
        self._accumulated_batches = 0

        if (res := self._handle_step_conditions()) is not None:
            return res

        self._log(
            event=TrainingEvent.STEP_ENDED,
            trainer=self,
        )

    def _forward(self, x: Tensor, target: Tensor) -> Tensor:
        x = x.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)

        with torch.autocast(device_type=str(self.device), dtype=self.precision):
            logits = self.model(x)
            loss = self.loss_fn(
                logits.reshape(-1, logits.size(-1)), target.reshape(-1).long()
            )
        return loss

    def _backward(self, loss: Tensor, x_shape):
        (loss / self.grad_accumulation).backward()
        dataset_length = x_shape[0] * x_shape[1]

        self._accumulated_batches += 1
        self._tokens_seen += dataset_length
        self._tot_loss += loss.item() * dataset_length

    def _train_one_epoch(self) -> TrainLoopResult:
        for x, target in self.train_loader:
            loss = self._forward(x, target)
            self._backward(loss, x.shape)

            if (res := self._process_commands()) is not None:
                return res

            if (
                self._accumulated_batches >= self.grad_accumulation
                and (res := self.step()) is not None
            ):
                return res

        return TrainLoopResult.EPOCH_COMPLETED

    def train(self):
        self._current_epoch = 1
        self._current_step = 0
        self._key_listener.start()
        self._log(
            event=TrainingEvent.TRAIN_STARTED,
            trainer=self,
        )
        try:
            while True:
                match self._train_one_epoch():
                    case TrainLoopResult.MAX_STEPS_REACHED:
                        self._log(
                            event=TrainingEvent.TRAIN_COMPLETED,
                            reason=TrainLoopResult.MAX_STEPS_REACHED,
                            trainer=self,
                        )
                        break
                    case TrainLoopResult.USER_CANCELLED:
                        self._log(
                            event=TrainingEvent.TRAIN_COMPLETED,
                            reason=TrainLoopResult.USER_CANCELLED,
                            trainer=self,
                        )
                        break
                    case TrainLoopResult.EPOCH_COMPLETED:
                        self._log(
                            event=TrainingEvent.EPOCH_COMPLETED,
                            reason=TrainLoopResult.EPOCH_COMPLETED,
                            trainer=self,
                        )
                self._current_epoch += 1
        except Exception as e:
            self._log(
                event=TrainingEvent.ERROR_OCCURRED,
                trainer=self,
                error_type=type(e).__name__,
                error_msg=str(e),
            )
            raise e
        finally:
            self._key_listener.stop()
            self.loggers.close()

    @property
    def dtype(self):
        return self.model.dtype

    @property
    def device(self):
        return self.model.device

    @property
    def precision(self):
        return self._precision

    @property
    def max_steps(self):
        return self._max_steps

    @property
    def tokens_seen(self):
        return self._tokens_seen

    @property
    def log_interval(self):
        return self._log_interval

    @property
    def current_step(self):
        return self._current_step

    @property
    def current_epoch(self):
        return self._current_epoch

    @property
    def grad_clip_norm(self):
        return self._grad_clip_norm

    @property
    def grad_accumulation(self):
        return self._grad_accumulation

    @dtype.setter
    def dtype(self, value: str | torch.dtype):
        dev_utils.type_check(
            ("value", value, str | torch.dtype),
            func_name=f"{self.__class__.__name__}.model_dtype.setter()",
        )
        value = value if isinstance(value, torch.dtype) else nn_utils.load_dtype(value)
        self.model.to(dtype=value)
