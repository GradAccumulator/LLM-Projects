from .logger import Logger, TrainingEvent, EVENT_CATEGORY_MAP
from pathlib import Path
from datetime import datetime
from .. import dev_utils, nn_utils
from training.llm.transformer_trainer.commands import TrainLoopResult


class TxtLogger(Logger):
    def __init__(
        self,
        log_dir: str | Path,
        file_name: str,
        append_timestamp_to_filename: bool = True,
    ):
        dev_utils.type_check(
            ("log_dir", log_dir, str | Path),
            ("file_name", file_name, str),
            ("append_timestamp_to_filename", append_timestamp_to_filename, bool),
            func_name="TxtLogger.__init__()",
        )
        if isinstance(log_dir, str):
            log_dir = Path(log_dir)
        if append_timestamp_to_filename:
            file_name += "_" + datetime.now().strftime("%Y%m%d_%H%M%S")

        self._log_dir = log_dir
        self._file_name = file_name
        self._path = log_dir / f"{file_name}.txt"

        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        now,
        trainer,
        event: TrainingEvent,
        loss=None,
        tokens_seen=None,
        reason: TrainLoopResult = None,
        ppl=None,
        error_name=None,
        error_msg=None,
        *_,
        **__,
    ):
        dt = now.strftime("%Y-%m-%dT%H:%M:%S")
        category = EVENT_CATEGORY_MAP[event].value
        with self._path.open(mode="a", encoding="utf-8") as f:
            record = f"[{dt}] [{category}] "
            match event:
                case TrainingEvent.VALIDATION_ENDED:
                    record += (
                        f"[{trainer.current_step}/{trainer.max_steps}] steps, "
                        f"loss= {loss:.6g}, ppl= {ppl:.6g}"
                    )
                case TrainingEvent.LOG_INTERVAL_REACHED:
                    record += (
                        f"{trainer.current_epoch} epochs, [{trainer.current_step}/{trainer.max_steps}] steps, "
                        f"loss= {loss:.6g}, tokens_seen= {tokens_seen}, lr= {trainer.optim.param_groups[0]["lr"]:.4g}"
                    )
                # SYSTEM
                case TrainingEvent.TRAIN_STARTED:
                    record += f"event=TRAIN_STARTED"
                case TrainingEvent.TRAIN_STOPPED:
                    record += f"event=TRAIN_STOPPED, step= {trainer.current_step}, reason= user_request"
                case TrainingEvent.TRAIN_RESUMED:
                    record += f"event=TRAIN_RESUMED, step= {trainer.current_step}, reason= user_request"
                case TrainingEvent.TRAIN_COMPLETED:
                    match reason:
                        case TrainLoopResult.USER_CANCELLED:
                            record += f"event=TRAIN_COMPLETED, step= {trainer.current_step}, reason= user_request"
                        case TrainLoopResult.MAX_STEPS_REACHED:
                            record += f"event=TRAIN_COMPLETED, step= {trainer.current_step}, reason= max_steps_reached"
                case TrainingEvent.EPOCH_COMPLETED:
                    record += f"event=EPOCH_COMPLETED, step= {trainer.current_step}, reason= epoch_completed"
                case TrainingEvent.ERROR_OCCURRED:
                    record += (
                        f"event=ERROR_OCCURRED, step= {trainer.current_step}, "
                        f"error_type= {error_name}, error_msg= {error_msg}"
                    )
                case _:
                    return
            f.write(record + "\n")

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def file_name(self):
        return self._file_name

    @property
    def path(self):
        return self._path
