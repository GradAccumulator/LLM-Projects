from abc import ABC, abstractmethod
from enum import Enum, auto


class TrainLoopResult(Enum):
    USER_CANCELLED = auto()
    EPOCH_COMPLETED = auto()
    MAX_STEPS_REACHED = auto()


class TrainingCommand(ABC):
    @abstractmethod
    def execute(self, trainer) -> TrainLoopResult | None: ...


class CancelTrainingCommand(TrainingCommand):
    def execute(self, trainer):
        return trainer.cancel_training()


class ValidateCommand(TrainingCommand):
    def execute(self, trainer):
        return trainer.validate()


class BreakpointCommand(TrainingCommand):
    def execute(self, trainer):
        return trainer.stop_train()

class SaveModelCommand(TrainingCommand):
    def execute(self, trainer):
        return trainer.save_model()