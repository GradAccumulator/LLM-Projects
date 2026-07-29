import json
from abc import ABC, abstractmethod
from pathlib import Path
from threading import Thread, Event, Lock
from datetime import datetime
from .. import dev_utils, nn_utils
from enum import StrEnum
from enum import Enum, auto
import torch


class TrainingEvent(Enum):
    TRAIN_STARTED = auto()
    TRAIN_RESUMED = auto()
    LOG_INTERVAL_REACHED = auto()
    TRAIN_STOPPED = auto()
    VALIDATION_ENDED = auto()
    TRAIN_COMPLETED = auto()
    ERROR_OCCURRED = auto()
    STEP_ENDED = auto()
    EPOCH_COMPLETED = auto()


class LogCategory(StrEnum):
    TRAIN = "TRAIN"
    VALIDATION = "VAL"
    SYSTEM = "SYSTEM"
    ERROR = "ERROR"


EVENT_CATEGORY_MAP = {
    TrainingEvent.LOG_INTERVAL_REACHED: LogCategory.TRAIN,
    TrainingEvent.STEP_ENDED: LogCategory.TRAIN,
    TrainingEvent.VALIDATION_ENDED: LogCategory.VALIDATION,
    TrainingEvent.TRAIN_STARTED: LogCategory.SYSTEM,
    TrainingEvent.TRAIN_RESUMED: LogCategory.SYSTEM,
    TrainingEvent.TRAIN_STOPPED: LogCategory.SYSTEM,
    TrainingEvent.TRAIN_COMPLETED: LogCategory.SYSTEM,
    TrainingEvent.EPOCH_COMPLETED: LogCategory.SYSTEM,
    TrainingEvent.ERROR_OCCURRED: LogCategory.ERROR,
}


class Logger(ABC):
    @abstractmethod
    def log(self, now, *args, **kwargs): ...

    def close(self):
        return

    def __call__(self, *args, **kwargs):
        kwargs["now"] = datetime.now()
        Thread(target=self.log, args=args, kwargs=kwargs, daemon=False).start()


class LoggerList(Logger):
    def __init__(self, *loggers):
        for i, logger in enumerate(loggers):
            dev_utils.type_check(
                (f"loggers[{i}]", logger, Logger), func_name="LoggerList.__init__()"
            )
        self._loggers: tuple[Logger] = loggers

    def log(self, *args, **kwargs):
        for logger in self.loggers:
            logger(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.log(*args, **kwargs)

    def close(self):
        for logger in self.loggers:
            logger.close()

    @property
    def loggers(self):
        return self._loggers
