from .pass_trainer_joint import T2IReIDTrainer


class TrainerFactory:
    @staticmethod
    def create(model_name, model, task_info, num_classes, is_distributed=False):
        if model_name == 'T2IReIDModel':
            return T2IReIDTrainer(model, task_info)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
