# D:\Instruct-ReID\reid\trainer\__init__.py

from .pass_trainer_joint import PASS_Trainer_Joint


class TrainerFactory:
    @staticmethod
    def create(model_name, model, task_info, num_classes, is_distributed=False):
        if model_name == 'PASS_Transformer_DualAttn_joint':
            # 只传递 model 和 task_info（这里 task_info 是 args）
            return PASS_Trainer_Joint(model, task_info)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
