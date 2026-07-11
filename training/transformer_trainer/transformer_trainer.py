import torch, torch.nn as nn
from torch import Tensor
from models import Transformer
import torch.utils.data as data
from utils import dev_utils,nn_utils
from queue import Queue, Empty
from . import commands
from .keyboard_listener import KeyboardListener
from .commands import TrainLoopResult, TrainingCommand

class TransformerTrainer:
    def __init__(
        self,
        model:Transformer,
        dataset:tuple[data.DataLoader,data.DataLoader],
        optimizer:torch.optim.Optimizer,
        scheduler:torch.optim.lr_scheduler.LRScheduler,
        loss_fn:nn.Module,
        log_interval:int,
        max_steps:int,
        precision:str='fp32',
        grad_clip_norm:float|int=1.0,
        grad_accumulation:int=1,
        validation_interval:int|None = None,
        validation:bool = True,
    ):
        self.model = model
        self.train_loader = dataset[0]
        self.valset_loader = dataset[1]
        self.optim = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn

        self._log_interval = log_interval
        self._precision = nn_utils.load_dtype(precision)
        self._max_steps = max_steps
        self._grad_clip_norm = grad_clip_norm
        self.validation_interval = validation_interval
        self.validation = validation
        self._grad_accumulation = grad_accumulation
        
        self._tot_loss = 0.0
        self._tot_dataset_length = 0
        self._accumulated_batches = 0
        self._command_queue:Queue[TrainingCommand] = Queue()
        self._current_step = 0
        self._key_listener = KeyboardListener(
            key_map={
                'q' : commands.CancelTrainingCommand,
                'b' : commands.BreakpointCommand,
                'v' : commands.ValidateCommand,
            },
            trainer=self
        )

    def submit_command(self, command):
        self._command_queue.put(command)
    
    def _process_commands(self) -> TrainLoopResult|None:
        while True:
            try:
                command = self._command_queue.get_nowait()
            except Empty:
                break
            
            res = command.execute(self)
            if res is not None:
                return res
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        tot_loss = 0.0
        total_dataset_length = 0

        for x,target in self.valset_loader:
            loss = self._forward(x, target)

            dataset_length = x.size(0) * x.size(1)
            tot_loss += loss*dataset_length
            total_dataset_length += dataset_length
        
        avg_loss = tot_loss.item()/total_dataset_length
        print(f"validated!! loss= {avg_loss}")
        self.model.train()
    
    def _log(self):
        self._tot_loss = float(self._tot_loss)
        print(
            f"[{self.current_step}/{self.max_steps}] steps, loss= {self._tot_loss/self._tot_dataset_length:.6g},"
            f" lr= {self.optim.param_groups[0]["lr"]:.3g}"
        )
        self._tot_loss = 0.0
        self._tot_dataset_length = 0
    
    def _handle_step_conditions(self):
        if (
            self.validation
            and self.validation_interval is not None
            and self.current_step % self.validation_interval == 0
        ):
            self.validate()

        if self.current_step%self.log_interval == 0:
            self._log()
        
        if self.current_step >= self.max_steps:
            return TrainLoopResult.MAX_STEPS_REACHED
    
    def _forward(self, x:Tensor, target:Tensor) -> Tensor:
        x = x.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)

        with torch.autocast(device_type=self.device, dtype=self.precision):
            logits = self.model(x)
            loss = self.loss_fn(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1)
            )
        return loss
    
    def _backward(self, loss:Tensor, x_shape):
        (loss/self.grad_accumulation).backward()
        dataset_length = x_shape[0] * x_shape[1]

        self._tot_loss += loss.item() * dataset_length
        self._tot_dataset_length += dataset_length
        self._accumulated_batches += 1

    def step(self):
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optim.step()
        self.scheduler.step()
        self.optim.zero_grad(set_to_none=True)
        
        self._current_step += 1
        self._accumulated_batches = 0
        
        if (res:=self._handle_step_conditions()) is not None:
            return res
    
    def _train_one_epoch(self) -> TrainLoopResult:
        for x,target in self.train_loader:
            loss = self._forward(x, target)
            self._backward(loss, x.shape)
            if (res:=self._process_commands()) is not None:
                return res
            if self._accumulated_batches >= self.grad_accumulation:
                res = self.step()
                if res is not None:
                    return res
        return TrainLoopResult.ONE_EPOCH_COMPLETED
    
    def train(self):
        epoch = 0
        self._key_listener.start()
        try:
            while True:
                match self._train_one_epoch():
                    case TrainLoopResult.MAX_STEPS_REACHED:
                        print(f"현재 step= {self.current_step}이 최대 step= {self.max_steps}에 도달하여 학습이 종료됩니다.")
                        break
                    case TrainLoopResult.USER_CANCELLED:
                        print("학습 종료")
                        break
                    case TrainLoopResult.ONE_EPOCH_COMPLETED:
                        print(f"현재 epoch= {epoch+1}")
                        print(f"{self._current_step}/{self.max_steps}스텝 학습 완료")
                epoch += 1
        finally:
            self._key_listener.stop()

    
    @property
    def log_interval(self): return self._log_interval
    @property
    def device(self): return self.model.device
    @property
    def precision(self): return self._precision
    @property
    def max_steps(self): return self._max_steps
    @property
    def grad_clip_norm(self): return self._grad_clip_norm
    @property
    def dtype(self): return self.model.dtype
    @dtype.setter
    def dtype(self,value:str|torch.dtype):
        dev_utils.type_check(
            ("value", value, str|torch.dtype),
            func_name=f"{self.__class__.__name__}.model_dtype.setter()"
        )
        value = value if isinstance(value, torch.dtype) else nn_utils.load_dtype(value)
        self.model.to(dtype=value)
    @property
    def current_step(self): return self._current_step
    @property
    def grad_accumulation(self): return self._grad_accumulation

