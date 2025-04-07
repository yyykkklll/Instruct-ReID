import argparse
import ast
import gc
import os
import random
import sys

import torch
import yaml
from torch.backends import cudnn

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

from reid.models.pass_transformer_joint import T2IReIDModel
from reid.datasets.data_builder_t2i import DataBuilder_t2i
from reid.trainer.pass_trainer_joint import T2IReIDTrainer
from reid.utils.logging import Logger
from reid.utils.lr_scheduler import WarmupMultiStepLR


def configuration():
    parser = argparse.ArgumentParser(description="Train T2I-ReID model")
    parser.add_argument('--config', default='scripts/config_cuhk_pedes.yaml', help='Path to config file')
    parser.add_argument('--root', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--dataset-configs', nargs='+', type=str, help='List of dataset configurations in JSON format')
    parser.add_argument('-b', '--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('-j', '--workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--height', type=int, default=224, help='Image height')
    parser.add_argument('--width', type=int, default=224, help='Image width')
    parser.add_argument('--lr', type=float, default=0.00035, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--warmup-step', type=int, default=1000, help='Warmup steps')
    parser.add_argument('--milestones', nargs='+', type=int, default=[7, 14], help='Milestones for LR scheduler')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--print-freq', type=int, default=50, help='Print frequency')
    parser.add_argument('--save-freq', type=int, default=5, help='Save frequency')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training')
    parser.add_argument('--bert-base-path', type=str, default='reid/bert-base-uncased', help='Path to BERT model')
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--logs-dir', type=str, default=os.path.join(working_dir, 'logs'), help='Directory for logs')

    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in config.items():
        if not hasattr(args, k) or getattr(args, k) == parser.get_default(k):
            setattr(args, k, v)

    # 解析 dataset_configs
    if args.dataset_configs:
        args.dataset_configs = [ast.literal_eval(cfg) for cfg in args.dataset_configs]
    else:
        raise ValueError("Must provide --dataset-configs for dataset configuration")

    args.img_size = (args.height, args.width)
    args.task_name = 't2i'
    return args


class Runner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = torch.amp.GradScaler('cuda', enabled=args.fp16)

    def build_optimizer(self, model):
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        return optimizer

    def build_scheduler(self, optimizer):
        return WarmupMultiStepLR(
            optimizer, self.args.milestones, gamma=0.1,
            warmup_factor=0.01, warmup_iters=self.args.warmup_step
        )

    def run(self):
        args = self.args

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # 日志
        os.makedirs(args.logs_dir, exist_ok=True)
        sys.stdout = Logger(os.path.join(args.logs_dir, 'log.txt'))
        print("==========\nArgs:{}\n==========".format(args))

        # 数据加载
        data_builder = DataBuilder_t2i(args, is_distributed=False)
        train_loader, _ = data_builder.build_data(is_train=True)
        query_loader, gallery_loader = data_builder.build_data(is_train=False)

        # 模型
        model = T2IReIDModel(net_config=args)
        model = model.to(self.device)

        # 优化器和调度器
        optimizer = self.build_optimizer(model)
        lr_scheduler = self.build_scheduler(optimizer)

        # 训练器
        trainer = T2IReIDTrainer(model, args)
        trainer.train(train_loader, optimizer, lr_scheduler)

        # 评估
        from reid.evaluation.evaluators_t import Evaluator_t2i
        evaluator = Evaluator_t2i(model)
        metrics = evaluator.evaluate(query_loader, gallery_loader, query_loader.dataset.data,
                                     gallery_loader.dataset.data)
        print("Evaluation Results:")
        print(metrics)


if __name__ == '__main__':
    cfg = configuration()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    runner = Runner(cfg)
    runner.run()
