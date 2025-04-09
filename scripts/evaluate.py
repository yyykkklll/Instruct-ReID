import argparse
import os
import sys
from pathlib import Path

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
    parser.add_argument('--config', default=os.path.join(ROOT_DIR, 'configs', 'config_cuhk_pedes.yaml'), help='Path to config file')
    parser.add_argument('--root', type=str, default=os.path.join(ROOT_DIR, 'data'), help='Root directory of the dataset')
    parser.add_argument('--dataset-configs', nargs='+', type=str, required=True,
                        help='Dataset configurations in JSON format')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for evaluation')
    parser.add_argument('--workers', type=int, default=0, help='Number of data loading workers')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision evaluation')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in config.items():
        if not hasattr(args, k) or getattr(args, k) == parser.get_default(k):
            setattr(args, k, v)

    return args


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.dataset_configs = [eval(cfg) for cfg in args.dataset_configs]

    data_builder = DataBuilder_t2i(args, is_distributed=False)
    query_loader, gallery_loader = data_builder.build_data(is_train=False)

    # 调试信息：检查数据加载
    print(f"Query data size: {len(query_loader.dataset.data)}, sample: {query_loader.dataset.data[:5]}")
    print(f"Gallery data size: {len(gallery_loader.dataset.data)}, sample: {gallery_loader.dataset.data[:5]}")

    model = T2IReIDModel(net_config=args)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # 兼容不同