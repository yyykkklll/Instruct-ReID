import argparse
import ast
import gc
import logging
import random
import sys
from pathlib import Path

import torch
import yaml
from torch.backends import cudnn

# 定义项目根目录并添加到 sys.path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

<<<<<<< HEAD
=======
# 导入 src 模块
>>>>>>> ae1d583f71d5b97df29d9414fb60417d2714e12b
from src.utils.serialization import save_checkpoint
from src.models.pass_transformer_joint import T2IReIDModel
from src.datasets.data_builder_t2i import DataBuilder_t2i
from src.trainer.pass_trainer_joint import T2IReIDTrainer
from src.utils.lr_scheduler import WarmupMultiStepLR


def configuration():
    """
    解析命令行参数并加载 YAML 配置文件
<<<<<<< HEAD
=======

    Returns:
        tuple: (args, config)
            - args: 命令行参数 (Namespace)
            - config: YAML 配置文件内容 (dict)
>>>>>>> ae1d583f71d5b97df29d9414fb60417d2714e12b
    """
    parser = argparse.ArgumentParser(description="Train T2I-ReID model")
    parser.add_argument('--config', default=str(ROOT_DIR / 'configs' / 'config_cuhk_pedes.yaml'),
                        help='Path to config file')
    parser.add_argument('--root', type=str, default=str(ROOT_DIR / 'data'),
                        help='Root directory of the dataset')
    parser.add_argument('--dataset-configs', nargs='+', type=str, help='List of dataset configurations in JSON format')
    parser.add_argument('-b', '--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('-j', '--workers', type=int, default=0, help='Number of data loading workers')
    parser.add_argument('--height', type=int, default=224, help='Image height')
    parser.add_argument('--width', type=int, default=224, help='Image width')
    parser.add_argument('--lr', type=float, default=0.003, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--warmup-step', type=int, default=1000, help='Warmup steps')
    parser.add_argument('--milestones', nargs='+', type=int, default=[10, 15], help='Milestones for LR scheduler')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--print-freq', type=int, default=50, help='Print frequency')
    parser.add_argument('--save-freq', type=int, default=5, help='Save frequency')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training')
    parser.add_argument('--bert-base-path', type=str, default=str(ROOT_DIR / 'pretrained' / 'bert-base-uncased'),
                        help='Path to BERT model')
    parser.add_argument('--vit-pretrained', type=str, default=str(ROOT_DIR / 'pretrained' / 'vit-base-patch16-224'),
                        help='Path to ViT model')
    parser.add_argument('--logs-dir', type=str, default=str(ROOT_DIR / 'logs'), help='Directory for logs')
    args = parser.parse_args()

    # 加载 YAML 配置文件
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with config_path.open('r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 将配置文件中的参数覆盖命令行默认值
    for k, v in config.items():
        if not hasattr(args, k) or getattr(args, k) == parser.get_default(k):
            setattr(args, k, v)

    # 处理数据集配置
    if args.dataset_configs:
        dataset_configs = []
        for cfg in args.dataset_configs:
            parsed = ast.literal_eval(cfg)
            if isinstance(parsed, list):
                dataset_configs.extend(parsed)
            else:
                dataset_configs.append(parsed)
        args.dataset_configs = dataset_configs
    else:
        args.dataset_configs = [
            {
                'name': 'CUHK-PEDES',
                'root': str(ROOT_DIR / 'data' / 'CUHK-PEDES'),
                'json_file': str(ROOT_DIR / 'data' / 'CUHK-PEDES' / 'annotations' / 'caption_all.json'),
                'cloth_json': str(ROOT_DIR / 'data' / 'CUHK-PEDES' / 'annotations' / 'caption_cloth.json'),
                'id_json': str(ROOT_DIR / 'data' / 'CUHK-PEDES' / 'annotations' / 'caption_id.json')
            },
            {
                'name': 'ICFG-PEDES',
                'root': str(ROOT_DIR / 'data' / 'ICFG-PEDES'),
                'json_file': str(ROOT_DIR / 'data' / 'ICFG-PEDES' / 'annotations' / 'ICFG-PEDES.json'),
                'cloth_json': str(ROOT_DIR / 'data' / 'ICFG-PEDES' / 'annotations' / 'caption_cloth.json'),
                'id_json': str(ROOT_DIR / 'data' / 'ICFG-PEDES' / 'annotations' / 'caption_id.json')
            },
            {
                'name': 'RSTPReid',
                'root': str(ROOT_DIR / 'data' / 'RSTPReid'),
                'json_file': str(ROOT_DIR / 'data' / 'RSTPReid' / 'annotations' / 'data_captions.json'),
                'cloth_json': str(ROOT_DIR / 'data' / 'RSTPReid' / 'annotations' / 'caption_cloth.json'),
                'id_json': str(ROOT_DIR / 'data' / 'RSTPReid' / 'annotations' / 'caption_id.json')
            }
        ]

<<<<<<< HEAD
    # 确保所有路径使用 Path 对象
=======
    # 确保所有路径使用 Path 对象，保持跨平台兼容性
>>>>>>> ae1d583f71d5b97df29d9414fb60417d2714e12b
    args.bert_base_path = str(Path(args.bert_base_path))
    args.vit_pretrained = str(Path(args.vit_pretrained))
    args.logs_dir = str(Path(args.logs_dir))
    args.root = str(Path(args.root))

    # 验证路径有效性
    if not Path(args.bert_base_path).exists():
        raise FileNotFoundError(f"BERT base path not found at: {args.bert_base_path}")
    if not Path(args.vit_pretrained).exists():
        raise FileNotFoundError(f"ViT base path not found at: {args.vit_pretrained}")

    args.img_size = (args.height, args.width)
    args.task_name = 't2i'
    return args, config


class Runner:
    """
    运行类，管理 T2I-ReID 模型的训练和评估
    """
<<<<<<< HEAD
    def __init__(self, args, config):
=======

    def __init__(self, args, config):
        """
        初始化 Runner，设置设备和混合精度训练

        Args:
            args: 命令行参数 (Namespace)
            config: YAML 配置文件内容 (dict)
        """
>>>>>>> ae1d583f71d5b97df29d9414fb60417d2714e12b
        self.args = args
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = torch.amp.GradScaler('cuda', enabled=args.fp16)

    def build_optimizer(self, model):
        """
        构建优化器
        """
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        return optimizer

    def build_scheduler(self, optimizer):
<<<<<<< HEAD
        if self.args.scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.args.epochs, eta_min=1e-6
            )
        else:
            return WarmupMultiStepLR(
                optimizer, self.args.milestones, gamma=0.1,
                warmup_factor=0.1, warmup_iters=self.args.warmup_step
            )

    def load_param(self, model, trained_path):
        param_dict = torch.load(trained_path, map_location=self.device, weights_only=True)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        elif 'model' in param_dict:
            param_dict = param_dict['model']
        model_dict = model.state_dict()
        for i in param_dict:
            if i in model_dict and model_dict[i].shape == param_dict[i].shape:
                model_dict[i] = param_dict[i]
        model.load_state_dict(model_dict, strict=False)

    def run(self):
=======
        """
        构建学习率调度器
        """
        return WarmupMultiStepLR(
            optimizer, self.args.milestones, gamma=0.1,
            warmup_factor=0.1, warmup_iters=self.args.warmup_step
        )

    def load_param(self, model, trained_path):
        """
        加载模型参数
        """
        param_dict = torch.load(trained_path, map_location=self.device, weights_only=True)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        elif 'model' in param_dict:
            param_dict = param_dict['model']
        model_dict = model.state_dict()
        for i in param_dict:
            if i in model_dict and model_dict[i].shape == param_dict[i].shape:
                model_dict[i] = param_dict[i]
        model.load_state_dict(model_dict, strict=False)

    def run(self):
        """
        执行训练和评估流程
        """
>>>>>>> ae1d583f71d5b97df29d9414fb60417d2714e12b
        args = self.args
        config = self.config
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # 创建日志目录
        Path(args.logs_dir).mkdir(parents=True, exist_ok=True)
        log_file = Path(args.logs_dir) / 'log.txt'
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s', 
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler(sys.stdout)
            ]
        )
<<<<<<< HEAD
=======
        logging.info(f"Args: {vars(args)}")
>>>>>>> ae1d583f71d5b97df29d9414fb60417d2714e12b

        # 构建数据集
        data_builder = DataBuilder_t2i(args, is_distributed=False)
        args.num_classes = data_builder.get_num_classes()
        config['model']['num_classes'] = args.num_classes
<<<<<<< HEAD
        logging.info(f"Set num_classes = {args.num_classes}")
=======
        logging.info(f"Set config.model.num_classes = {args.num_classes}")
>>>>>>> ae1d583f71d5b97df29d9414fb60417d2714e12b

        train_loader, _ = data_builder.build_data(is_train=True)
        query_loader, gallery_loader = data_builder.build_data(is_train=False)
        logging.info(f"Train data size: {len(train_loader.dataset.data)}")
        logging.info(f"Query data size: {len(query_loader.dataset.data)}")

        # 初始化模型
        model = T2IReIDModel(net_config=config.get('model', {}))
        model = model.to(self.device)

        # 构建优化器和调度器
        optimizer = self.build_optimizer(model)
        lr_scheduler = self.build_scheduler(optimizer)

        # 初始化训练器
        trainer = T2IReIDTrainer(model, args)
        trainer.train(
            train_loader, optimizer, lr_scheduler, query_loader, gallery_loader, checkpoint_dir=args.logs_dir
        )

        # 保存最终检查点
        checkpoint_path = Path(args.logs_dir) / 'checkpoint_epoch_final.pth'
        save_checkpoint({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': args.epochs
        }, fpath=str(checkpoint_path))
        logging.info(f"Model saved at: {checkpoint_path}")

        # 评估模型
        from src.evaluation.evaluators_t import Evaluator_t2i
        self.load_param(model, str(checkpoint_path))
        evaluator = Evaluator_t2i(model, args=args)
        metrics = evaluator.evaluate(
            query_loader, gallery_loader, query_loader.dataset.data,
            gallery_loader.dataset.data, checkpoint_path=str(checkpoint_path)
        )
        logging.info(f"Evaluation Results: {metrics}")


if __name__ == '__main__':
    args, config = configuration()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    runner = Runner(args, config)
<<<<<<< HEAD
    runner.run()
=======
    runner.run()
>>>>>>> ae1d583f71d5b97df29d9414fb60417d2714e12b
