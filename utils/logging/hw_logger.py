from .logger import Logger, TrainingEvent
from pathlib import Path
from threading import Thread, Event, Lock
from datetime import datetime
from .. import dev_utils,nn_utils
import json
import psutil
import pynvml as nvml
import torch

MIB = 1024 ** 2
GIB = 1024 ** 3

class HWLogger(Logger):
    def __init__(
        self,
        log_dir:str|Path,
        file_name:str,
        interval: float = 5.0,
        append_timestamp_to_filename:bool=True,
        cuda_idx = 0,
    ):
        dev_utils.type_check(
            ("log_dir", log_dir, str|Path),
            ("file_name", file_name, str),
            ("append_timestamp_to_filename", append_timestamp_to_filename, bool),
            ("interval", interval, int | float),
            ("cuda_idx", cuda_idx, int),
            func_name="JsonlLogger.__init__()"
        )
        if isinstance(log_dir, str):
            log_dir = Path(log_dir)
        if append_timestamp_to_filename:
            file_name += "_"+datetime.now().strftime("%Y%m%d_%H%M%S")
        
        nvml.nvmlInit()
        
        self._log_dir = log_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._file_name = file_name
        self._interval = interval
        self._cuda_idx = cuda_idx

        self._path = log_dir/f"{file_name}.jsonl"
        self._gpu_handle = nvml.nvmlDeviceGetHandleByIndex(self.cuda_idx)
        self._stop_event = Event()
        self._thread = None
        self._current_step = 0
        self._step_lock = Lock()
        self._device = torch.device(self.cuda_idx)

        self._stop_event.set()
        psutil.cpu_percent(interval=None)
    
    def _get_state(self):
        vram = nvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
        reason_mask = nvml.nvmlDeviceGetCurrentClocksEventReasons(self._gpu_handle)
        memory = psutil.virtual_memory()
        cuda_stats = torch.cuda.memory_stats(self._device)
        return {
            #nvml
            "vram_used_mib":vram.used/MIB,
            "vram_used_percent":vram.used/vram.total*100,
            "gpu_temperature_c":nvml.nvmlDeviceGetTemperature(self._gpu_handle, nvml.NVML_TEMPERATURE_GPU),
            "gpu_power_w":nvml.nvmlDeviceGetPowerUsage(self._gpu_handle)/1000,
            "power_limit_w":nvml.nvmlDeviceGetPowerManagementLimit(self._gpu_handle)/1000,
            "gpu_sm_clock_mhz":nvml.nvmlDeviceGetClockInfo(self._gpu_handle, nvml.NVML_CLOCK_SM),
            "gpu_utilization":nvml.nvmlDeviceGetUtilizationRates(self._gpu_handle).gpu,
            "gpu_memory_clock_mhz":nvml.nvmlDeviceGetClockInfo(self._gpu_handle, nvml.NVML_CLOCK_MEM),
            "limit_idle":bool(reason_mask & nvml.nvmlClocksEventReasonGpuIdle),
            "limit_power":bool(reason_mask & nvml.nvmlClocksEventReasonSwPowerCap),
            "limit_sw_thermal":bool(reason_mask & nvml.nvmlClocksEventReasonSwThermalSlowdown),
            "limit_hw_thermal":bool(reason_mask & nvml.nvmlClocksEventReasonHwThermalSlowdown),
            "limit_power_brake":bool(reason_mask & nvml.nvmlClocksEventReasonHwPowerBrakeSlowdown),
            #psutil
            "cpu_utilization":psutil.cpu_percent(),
            "ram_available_mib":memory.available / MIB,
            "ram_used_mib": (memory.total - memory.available)/MIB,
            "ram_utilization_percent": memory.percent,
            #cuda
            "cuda_allocated_mib": cuda_stats["allocated_bytes.all.current"]/MIB,
            "cuda_reserved_mib":cuda_stats["reserved_bytes.all.current"]/MIB,
            "cuda_peak_allocated_mib":cuda_stats["allocated_bytes.all.peak"]/MIB,
            "cuda_peak_reserved_mib":cuda_stats["reserved_bytes.all.peak"]/MIB,
            "cuda_allocation_retries":cuda_stats["num_alloc_retries"],
            "cuda_oom_count": cuda_stats["num_ooms"],
        }

    def _run(self):
        with self.path.open(mode='a', encoding='utf-8') as f:
            while not self._stop_event.is_set():
                record = {
                    "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                    "step": self._get_current_step(),
                    **self._get_state(),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
                self._stop_event.wait(self.interval)
    
    def start(self):
        if not self._stop_event.is_set():
            return
        
        self._stop_event.clear()

        self._thread = Thread(target=self._run)
        self._thread.start()

    def stop(self):
        if self._stop_event.is_set():
            return

        self._stop_event.set()
        
        self._thread.join()
    
    def close(self):
        self.stop()
        nvml.nvmlShutdown()
    
    def update_step(self, step: int) -> None:
        if step < 0:
            raise ValueError(
                f"step은 0 이상이어야 합니다. 현재 값: {step}"
            )

        with self._step_lock:
            self._current_step = step
    
    def _get_current_step(self) -> int:
        with self._step_lock:
            return self._current_step
    
    def log(self, event: TrainingEvent, trainer, *_,**__):
        match event:
            case TrainingEvent.TRAIN_STARTED|TrainingEvent.TRAIN_RESUMED:
                self.start() 
            case TrainingEvent.STEP_ENDED:
                self.update_step(trainer.current_step)
            case TrainingEvent.TRAIN_STOPPED:
                if trainer.current_step is not None:
                    self.update_step(trainer.current_step)
                self.stop()
            case TrainingEvent.TRAIN_COMPLETED|TrainingEvent.ERROR_OCCURRED:
                if trainer.current_step is not None:
                    self.update_step(trainer.current_step)
                self.close()
            case _:
                return

    @property
    def log_dir(self): return self._log_dir
    @property
    def file_name(self): return self._file_name
    @property
    def path(self): return self._path
    @property
    def interval(self): return self._interval
    @property
    def cuda_idx(self): return self._cuda_idx