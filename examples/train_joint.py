import argparse
import gc
import os
import os.path as osp
import random
import sys

import torch
import torch.cuda.amp
import torch.utils.data
import yaml
from torch.backends import cudnn

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

from reid.models.pass_transformer_joint import T2IReIDModel
from reid.datasets.data_builder_t2i import DataBuilder_t2i
from reid.trainer.pass_trainer_joint import T2IReIDTrainer
from reid.utils.logging import Logger
from reid.utils.lr_scheduler import WarmupMultiStepLR
from reid.utils.distributed_utils_pt import dist_init_singletask

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"


def configuration():
    parser = argparse.ArgumentParser(description="Train T2I-ReID model with CUHK-PEDES dataset")
    parser.add_argument('--port', type=int, default=29500, help='port for distributed training')
    parser.add_argument('--config', default='scripts/config_cuhk_pedes.yaml')
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--train-list', type=str, required=True)
    parser.add_argument('--query-list', type=str, required=True)
    parser.add_argument('--gallery-list', type=str, required=True)
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--width', type=int, default=224)  # ViT 需要 224x224
    parser.add_argument('--local_rank', type=int, default=None)
    parser.add_argument('--lr', type=float, default=0.00035)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--warmup-step', type=int, default=1000)
    parser.add_argument('--milestones', nargs='+', type=int, default=[7000, 14000])
    parser.add_argument('--epochs', type=int, default=20)  # 替换 iters
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--save-freq', type=int, default=5)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--bert-base-path', type=str, default='bert-base-uncased')
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--logs-dir', type=str, default=osp.join(working_dir, 'logs'))
    parser.add_argument('--vit-pretrained', type=str, default='logs/pretrained/pass_vit_base_full.pth')

    args = parser.parse_args()
    command_line_args = vars(args).copy()

    with open(args.config, encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    args.data_config = {'json_file': config['json_file']}

    for k, v in config.items():
        if k not in command_line_args or command_line_args[k] == parser.get_default(k):
            setattr(args, k, v)

    args.local_rank = int(os.environ.get('LOCAL_RANK', -1)) if args.local_rank is None else args.local_rank
    args.root = os.path.abspath(args.root)
    args.train_list = os.path.abspath(args.train_list)
    args.query_list = os.path.abspath(args.query_list)
    args.gallery_list = os.path.abspath(args.gallery_list)
    args.logs_dir = os.path.abspath(args.logs_dir)
    args.vit_pretrained = os.path.abspath(args.vit_pretrained)
    args.config = os.path.abspath(args.config)
    args.data_config['json_file'] = os.path.abspath(args.data_config['json_file'])

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

    def distributed(self, model):
        model = model.to(self.device)
        if self.args.local_rank >= 0 and torch.cuda.device_count() > 1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank
            )
        return model

    def run(self):
        args = self.args

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        rank, world_size, is_distributed = dist_init_singletask(args) if args.local_rank >= 0 else (0, 1, False)
        self.is_distributed = is_distributed

        if rank == 0:
            os.makedirs(args.logs_dir, exist_ok=True)
            sys.stdout = Logger(os.path.join(args.logs_dir, 'log.txt'))
        print("==========\nArgs:{}\n==========".format(args))

        # 数据加载
        data_builder = DataBuilder_t2i(args, is_distributed=is_distributed)
        train_loader, train_set = data_builder.build_data(is_train=True)
        num_classes = len(set([item[2] for item in train_set.data]))
        if rank == 0:
            print(f"Number of classes (unique IDs): {num_classes}")

        # 模型
        model = T2IReIDModel(num_classes=num_classes, net_config=args)
        if os.path.exists(args.vit_pretrained):
            model.load_param(args.vit_pretrained)
        model = self.distributed(model)

        # 优化器和调度器
        optimizer = self.build_optimizer(model)
        lr_scheduler = self.build_scheduler(optimizer)

        # 训练器
        trainer = T2IReIDTrainer(model, args)
        trainer.scaler = self.scaler
        trainer.train(train_loader, optimizer, lr_scheduler)


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
