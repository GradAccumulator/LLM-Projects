from .logger import Logger, TrainingEvent
from training.llm.transformer_trainer.commands import TrainLoopResult
from .. import dev_utils


class TerminalLogger(Logger):
    def log(
        self,
        event,
        trainer,
        loss=None,
        ppl=None,
        reason: TrainLoopResult = None,
        *_,
        **__,
    ):
        match event:
            case TrainingEvent.LOG_INTERVAL_REACHED:
                print(
                    f"{trainer.current_epoch} epochs, [{trainer.current_step}/{trainer.max_steps}] steps, "
                    f"loss= {loss:.6g}, lr= {trainer.optim.param_groups[0]["lr"]:.3g}"
                )
            case TrainingEvent.VALIDATION_ENDED:
                print(f"validated!! loss= {loss}, ppl= {ppl}")
            case TrainingEvent.TRAIN_STARTED:
                model_params = sum(p.numel() for p in trainer.model.parameters())
                print(
                    f"\nmodel params: {model_params}(≈ {dev_utils.num_to_str(model_params)})"
                    f"\nmax steps: {trainer.max_steps}"
                    f"\ndataset_name: {trainer.train_loader.dataset.name}"
                    f"\noptimizer: {trainer.optim.__class__.__name__}"
                    f"\n{"-"*30}학습 시작{"-"*30}"
                )
            case TrainingEvent.TRAIN_COMPLETED:
                match reason:
                    case TrainLoopResult.MAX_STEPS_REACHED:
                        print(
                            f"현재 step= {trainer.current_step}이 최대 step= {trainer.max_steps}에 도달하여 학습이 종료됩니다."
                        )
                    case TrainLoopResult.USER_CANCELLED:
                        print("개발자의 요쳥으로 학습이 종료됩니다.")
            case TrainingEvent.EPOCH_COMPLETED:
                print(f"현재 epoch= {trainer.current_epoch}")
                print(f"{trainer.current_step}/{trainer.max_steps}스텝 학습 완료")
            case _:
                return
