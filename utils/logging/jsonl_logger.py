from .logger import Logger, TrainingEvent, EVENT_CATEGORY_MAP
import json
from pathlib import Path
from datetime import datetime
from .. import dev_utils, nn_utils
from training.llm.transformer_trainer.transformer_trainer import TrainLoopResult


class JsonlLogger(Logger):
    def __init__(
        self,
        log_dir: str | Path,
        file_name: str,
        append_timestamp_to_filename: bool = True,
    ):
        func_name = "JsonlLogger.__init__()"
        dev_utils.type_check(
            ("log_dir", log_dir, str | Path),
            ("file_name", file_name, str),
            ("append_timestamp_to_filename", append_timestamp_to_filename, bool),
            func_name=func_name,
        )
        if isinstance(log_dir, str):
            log_dir = Path(log_dir)
        if append_timestamp_to_filename:
            file_name += "_" + datetime.now().strftime("%Y%m%d_%H%M%S")

        self._log_dir = log_dir
        self._file_name = file_name
        self._path = log_dir / f"{file_name}.jsonl"

        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        now,
        trainer,
        event: TrainingEvent,
        loss=None,
        tokens_seen=None,
        ppl=None,
        error_type=None,
        error_msg=None,
        reason: TrainLoopResult = None,
        *_,
        **__,
    ):
        dt = now.strftime("%Y-%m-%dT%H:%M:%S")
        record = {
            "timestamp": dt,
            "category": EVENT_CATEGORY_MAP[event].value,
            "step": trainer.current_step,
        }
        match event:
            case TrainingEvent.VALIDATION_ENDED:
                record.update(
                    {
                        "loss": loss,
                        "ppl": ppl,
                    }
                )
            case TrainingEvent.LOG_INTERVAL_REACHED:
                record.update(
                    {
                        "epoch": trainer.current_epoch,
                        "loss": loss,
                        "tokens_seen": tokens_seen,
                        "lr": trainer.optim.param_groups[0]["lr"],
                    }
                )
            # SYSTEM
            case TrainingEvent.TRAIN_STARTED:
                record.update(
                    {
                        "event": "TRAIN_STARTED",
                    }
                )
            case TrainingEvent.TRAIN_STOPPED:
                record.update(
                    {
                        "event": "TRAIN_STOPPED",
                        "reason": "user_request",
                    }
                )
            case TrainingEvent.TRAIN_RESUMED:
                record.update(
                    {
                        "event": "TRAIN_RESUMED",
                        "reason": "user_request",
                    }
                )
            case TrainingEvent.TRAIN_COMPLETED:
                record.update(
                    {
                        "event": "TRAIN_COMPLETED",
                    }
                )
                match reason:
                    case TrainLoopResult.USER_CANCELLED:
                        record.update(
                            {
                                "reason": "user_request",
                            }
                        )
                    case TrainLoopResult.MAX_STEPS_REACHED:
                        record.update(
                            {
                                "reason": "max_steps_reached",
                            }
                        )
            case TrainingEvent.EPOCH_COMPLETED:
                record.update(
                    {
                        "event": "EPOCH_COMPLETED",
                        "reason": "one epoch completed",
                    }
                )
            case TrainingEvent.ERROR_OCCURRED:
                record.update(
                    {
                        "event": "ERROR_OCCURRED",
                        "error_type": error_type,
                        "error_msg": error_msg,
                    }
                )
            case _:
                return
        with self._path.open(mode="a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @property
    def path(self):
        return self._path

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def file_name(self):
        return self._file_name
