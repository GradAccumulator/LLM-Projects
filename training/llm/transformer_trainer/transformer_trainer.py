from torch      import Tensor
from queue      import Queue, Empty
from typing     import overload
from omegaconf  import DictConfig
import torch, torch.nn  as nn
import torch.utils.data as data

from .                  import commands
from models             import Transformer
from .keyboard_listener import KeyboardListener
from utils              import dev_utils,nn_utils
from .commands          import TrainLoopResult, TrainingCommand

class TransformerTrainer:
    @overload
    def __init__(
        self,
        model     :Transformer,
        dataset   :tuple[data.DataLoader,data.DataLoader],
        optimizer :torch.optim.Optimizer,
        scheduler :torch.optim.lr_scheduler.LRScheduler,
        loss_fn   :nn.Module,
        cfg       :DictConfig|dict
    ):...
    @overload
    def __init__(
        self,
        model               :Transformer,
        dataset             :tuple[data.DataLoader,data.DataLoader],
        optimizer           :torch.optim.Optimizer,
        scheduler           :torch.optim.lr_scheduler.LRScheduler,
        loss_fn             :nn.Module,
        log_interval        :int,
        max_steps           :int,
        precision           :str       ='fp32',
        grad_clip_norm      :float|int = 1.0,
        grad_accumulation   :int       = 1,
        validation_interval :int|None  = None,
        validation          :bool      = True,
    ):...
    def __init__(
        self,
        model               :Transformer,
        dataset             :tuple[data.DataLoader,data.DataLoader]|list[data.DataLoader,data.DataLoader],
        optimizer           :torch.optim.Optimizer,
        scheduler           :torch.optim.lr_scheduler.LRScheduler,
        loss_fn             :nn.Module,
        log_interval        :int|DictConfig|dict,
        max_steps           :int,
        precision           :str       ='fp32',
        grad_clip_norm      :float|int = 1.0,
        grad_accumulation   :int       = 1,
        validation_interval :int|None  = None,
        validation          :bool      = True,
        cfg                 :DictConfig|dict = None
    ):
        func_name = "TransformerTrainer.__init__()"
        dev_utils.type_check(
            ("loss_fn"    ,loss_fn     ,nn.Module),
            ("dataset"    ,dataset     ,tuple|list),
            ("model"      ,model       ,Transformer),
            ("dataset[0]" ,dataset[0]  ,data.DataLoader),
            ("dataset[1]" ,dataset[1]  ,data.DataLoader),
            ("optim"      ,optimizer   ,torch.optim.Optimizer),
            ("scheduler"  ,scheduler   ,torch.optim.lr_scheduler.LRScheduler),
            func_name=func_name
        )
        self.model          = model
        self.loss_fn        = loss_fn
        self.optim          = optimizer
        self.scheduler      = scheduler
        self.train_loader   = dataset[0]
        self.valset_loader  = dataset[1]
        
        if (
            isinstance(cfg, DictConfig|dict)
            or isinstance(log_interval, DictConfig|dict)
        ):
            if isinstance(cfg, DictConfig|dict):
                cfg = dev_utils.make_dictconfig(cfg)
            elif isinstance(log_interval, DictConfig|dict):
                cfg = dev_utils.make_dictconfig(log_interval)
            dev_utils.check_dictconfig(
                cfg,
                ["train.max_steps", "train.precision", "train.validation", "train.log_interval", "train.validation_interval", "optimizer.grad_clip_norm", "optimizer.grad_accumulation"]
            )
            max_steps           = cfg.train.max_steps
            precision           = cfg.train.precision
            validation          = cfg.train.validation
            log_interval        = cfg.train.log_interval
            validation_interval = cfg.train.validation_interval
            grad_clip_norm      = cfg.optimizer.grad_clip_norm
            grad_accumulation   = cfg.optimizer.grad_accumulation
        
        dev_utils.type_check(
            ("max_steps"           ,max_steps           ,int),
            ("log_interval"        ,log_interval        ,int),
            ("grad_accumulation"   ,grad_accumulation   ,int),
            ("validation_interval" ,validation_interval ,int),
            ("validation"          ,validation          ,bool),
            ("grad_clip_norm"      ,grad_clip_norm      ,float|int),
            ("precision"           ,precision           ,str|torch.dtype),
            func_name=func_name
        )

        self._max_steps      = max_steps
        self.validation      = validation
        self._log_interval   = log_interval
        self._grad_clip_norm = grad_clip_norm
        self._precision      = nn_utils.load_dtype(precision)
        self._grad_accumulation  = grad_accumulation
        self.validation_interval = validation_interval
        
        self._tot_loss = 0.0
        self._tot_dataset_length  = 0
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
            
            if (res:=command.execute(self)) is not None:
                return res
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        tot_loss = 0.0
        total_dataset_length = 0

        for x,target in self.valset_loader:
            loss = self._forward(x, target)

            dataset_length = x.size(0) * x.size(1)
            tot_loss       += loss*dataset_length
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
        self._tot_dataset_length = 0
        self._tot_loss           = 0.0
    
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

    def step(self):
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optim.step()
        self.scheduler.step()
        self.optim.zero_grad(set_to_none=True)
        
        self._current_step        += 1
        self._accumulated_batches = 0
        
        if (res:=self._handle_step_conditions()) is not None:
            return res
    
    def _forward(self, x:Tensor, target:Tensor) -> Tensor:
        x      = x.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)

        with torch.autocast(device_type=self.device, dtype=self.precision):
            logits = self.model(x)
            loss   = self.loss_fn(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1)
            )
        return loss
    
    def _backward(self, loss:Tensor, x_shape):
        (loss/self.grad_accumulation).backward()
        dataset_length = x_shape[0] * x_shape[1]

        self._accumulated_batches += 1
        self._tot_dataset_length  += dataset_length
        self._tot_loss += loss.item() * dataset_length
    
    def _train_one_epoch(self) -> TrainLoopResult:
        for x,target in self.train_loader:
            loss = self._forward(x, target)
            self._backward(loss, x.shape)
            
            if (res:=self._process_commands()) is not None:
                return res
            
            if (
                self._accumulated_batches >= self.grad_accumulation
                and (res:=self.step()) is not None
            ):
                return res
            
        return TrainLoopResult.ONE_EPOCH_COMPLETED
    
    def train(self):
        epoch = 1
        self._current_step = 0
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
                        print(f"현재 epoch= {epoch}")
                        print(f"{self._current_step}/{self.max_steps}스텝 학습 완료")
                epoch += 1
        finally:
            self._key_listener.stop()
    
    @property
    def dtype(self): return self.model.dtype
    @property
    def device(self): return self.model.device
    @property
    def precision(self): return self._precision
    @property
    def max_steps(self): return self._max_steps
    @property
    def log_interval(self): return self._log_interval
    @property
    def current_step(self): return self._current_step
    @property
    def grad_clip_norm(self): return self._grad_clip_norm
    @property
    def grad_accumulation(self): return self._grad_accumulation
    
    @dtype.setter
    def dtype(self,value:str|torch.dtype):
        dev_utils.type_check(
            ("value", value, str|torch.dtype),
            func_name=f"{self.__class__.__name__}.model_dtype.setter()"
        )
        value = value if isinstance(value, torch.dtype) else nn_utils.load_dtype(value)
        self.model.to(dtype=value)

