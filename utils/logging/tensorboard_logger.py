from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datetime import datetime

from .logger import Logger, TrainingEvent
from utils import dev_utils


class TensorboardLogger(Logger):
    def __init__(
        self,
        log_dir: str | Path,
        run_name: str,
        append_timestamp_to_filename: bool = True,
    ):
        func_name = "TensorboardLogger.__init__()"
        dev_utils.type_check(
            ("log_dir", log_dir, str | Path),
            ("run_name", run_name, str),
            ("append_timestamp_to_filename", append_timestamp_to_filename, bool),
            func_name=func_name,
        )
        if isinstance(log_dir, str):
            log_dir = Path(log_dir)
        if append_timestamp_to_filename:
            run_name += "_" + datetime.now().strftime("%Y%m%d_%H%M%S")

        self._log_dir = log_dir
        self._run_name = run_name
        self._path = log_dir / run_name

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path.mkdir(parents=True, exist_ok=True)

        self._writer = SummaryWriter(log_dir=self.path)

    def log(self, event: TrainingEvent, trainer, loss=None, ppl=None, *_, **__):
        match event:
            case TrainingEvent.STEP_ENDED:
                self.writer.add_scalar(
                    "train/lr",
                    trainer.optim.param_groups[0]["lr"],
                    trainer.current_step,
                )
            case TrainingEvent.VALIDATION_ENDED:
                self.writer.add_scalars(
                    "val", {"loss": loss, "ppl": ppl}, trainer.current_step
                )
            case TrainingEvent.LOG_INTERVAL_REACHED:
                self.writer.add_scalar(
                    "train/loss",
                    loss,
                    trainer.current_step
                )
            case _:
                return

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def path(self):
        return self._path

    @property
    def run_name(self):
        return self._run_name

    @property
    def writer(self):
        return self._writer

    def close(self):
        self.writer.close()
