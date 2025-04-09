import argparse
import os
import sys
from pathlib import Path
import logging

import torch
import yaml

# 添加项目根目录到 sys.path
ROOT_DIR = Path(__file__).parent.parent  # 从 scripts/ 到 v3/
sys.path.insert(0, str(ROOT_DIR))

from src.models.pass_transformer_joint import T2IReIDModel
from src.datasets.data_builder_t2i import DataBuilder_t2i
from src.evaluation.evaluators_t import Evaluator_t2i


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate T2I-ReID model")
    ROOT_DIR = Path(__file__).parent.parent  # 从 scripts/ 到 v3/
    parser.add_argument('--config', default=os.path.join(ROOT_DIR, 'configs', 'config_cuhk_pedes.yaml'), 
                        help='Path to config file')
    parser.add_argument('--root', type=str, default=os.path.join(ROOT_DIR, 'data'), 
                        help='Root directory of the dataset')
    parser.add_argument('--dataset-configs', nargs='+', type=str, required=True, 
                        help='Dataset configurations in JSON format')
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to trained model checkpoint')
    parser.add_argument('--batch-size', type=int, default=128, 
                        help='Batch size for evaluation')
    parser.add_argument('--workers', type=int, default=0, 
                        help='Number of data loading workers')
    parser.add_argument('--fp16', action='store_true', 
                        help='Use mixed precision evaluation')
    args = parser.parse_args()

    # 在解析参数后立即将 dataset_configs 转换为字典
    args.dataset_configs = [eval(cfg) for cfg in args.dataset_configs]

    with open(args.config, encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in config.items():
        if not hasattr(args, k) or getattr(args, k) == parser.get_default(k):
            setattr(args, k, v)

    return args


def setup_logging(args):
    # 从 dataset_configs 获取数据集名称
    dataset_name = args.dataset_configs[0]['name'].lower()
    # 日志目录
    log_dir = os.path.join(ROOT_DIR, 'logs', f"{dataset_name}_evaluate")
    os.makedirs(log_dir, exist_ok=True)
    # 日志文件名：检查点文件名 + "_log.txt"
    checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    log_file = os.path.join(log_dir, f"{checkpoint_name}_log.txt")

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)  # 同时输出到终端
        ]
    )
    return logging.getLogger(__name__)


def main():
    args = parse_args()
    logger = setup_logging(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构建数据加载器
    data_builder = DataBuilder_t2i(args, is_distributed=False)
    query_loader, gallery_loader = data_builder.build_data(is_train=False)

    # 日志记录数据加载信息
    logger.info(f"Query data size: {len(query_loader.dataset.data)}, sample: {query_loader.dataset.data[:5]}")
    logger.info(f"Gallery data size: {len(gallery_loader.dataset.data)}, sample: {gallery_loader.dataset.data[:5]}")

    # 初始化模型
    model = T2IReIDModel(net_config=args)
    model = model.to(device)

    # 加载检查点
    checkpoint = torch.load(args.checkpoint, map_location=device)
    loaded_layers = 0
    total_layers = sum(1 for _ in model.state_dict().items())
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint  # 假设检查点直接是 state_dict
    for name, param in state_dict.items():
        if name in model.state_dict():
            model.state_dict()[name].copy_(param)
            loaded_layers += 1
    logger.info(f"Loaded {loaded_layers} / {total_layers} layers from {args.checkpoint}")

    # 设置为评估模式
    model.eval()

    # 初始化评估器
    evaluator = Evaluator_t2i(model)

    # 准备查询和图库数据
    query = query_loader.dataset.data
    gallery = gallery_loader.dataset.data

    # 运行评估
    with torch.cuda.amp.autocast(enabled=args.fp16):
        results = evaluator.evaluate(query_loader, gallery_loader, query, gallery)

    # 记录最终结果
    logger.info("Final Evaluation Results:")
    logger.info(f"  mAP: {results['mAP']:.4f}")
    logger.info(f"  Rank-1: {results['rank1']:.4f}")
    logger.info(f"  Rank-5: {results['rank5']:.4f}")
    logger.info(f"  Rank-10: {results['rank10']:.4f}")


if __name__ == '__main__':
    main()