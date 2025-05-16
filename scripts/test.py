import argparse
import os
import sys
from pathlib import Path

import torch.nn as nn
import yaml

# 调整导入路径为相对导入
from src.datasets.data_builder_t2i import DataBuilder_t2i
from src.evaluation.evaluators_t import Evaluator_t2i
from src.models.pass_transformer_joint import T2IReIDModel
from src.utils.logging import Logger
from src.utils.serialization import load_checkpoint


def main_worker(args):
    # 项目根目录定位
    ROOT_DIR = Path(__file__).parent.parent  # 从 scripts/ 到 TextGuidedReID/
    
    log_dir = os.path.join(ROOT_DIR, "logs", os.path.dirname(args.resume).split(os.sep)[-1])
    checkpoint_name = os.path.basename(args.resume).split('.')[0]
    sys.stdout = Logger(os.path.join(log_dir, f'{checkpoint_name}_log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # 数据加载
    data_builder = DataBuilder_t2i(args)
    query_loader, gallery_loader = data_builder.build_data(is_train=False)

    # 打印数据加载信息
    print(f"Query dataset size: {len(query_loader.dataset)}")
    print(f"Gallery dataset size: {len(gallery_loader.dataset)}")
    for batch in query_loader:
        images, captions, pids, cam_ids = batch  # 解包元组
        print(f"Query batch shape - images: {images.shape}, pids: {pids.shape}, captions: {len(captions)} items")
        break
    for batch in gallery_loader:
        images, captions, pids, cam_ids = batch  # 解包元组
        print(f"Gallery batch shape - images: {images.shape}, pids: {pids.shape}, captions: {len(captions)} items")
        break

    # 检查查询和图库的样本路径是否有交集
    query_paths = set(item[0] for item in query_loader.dataset.data)
    gallery_paths = set(item[0] for item in gallery_loader.dataset.data)
    common_paths = query_paths.intersection(gallery_paths)
    print(f"Number of common image paths between query and gallery: {len(common_paths)}")
    if len(common_paths) > 0:
        print(f"Sample common paths (first 5): {list(common_paths)[:5]}")

    # 加载训练集（动态路径）
    args.train_list = os.path.join(ROOT_DIR, "data", "CUHK-PEDES", "splits", "train_t2i_v2.txt")  # 动态构建，避免硬编码
    data_builder = DataBuilder_t2i(args)  # 重新创建 data_builder，确保 train_list 被设置
    train_loader, _ = data_builder.build_data(is_train=True)  # 解包元组，忽略 val_loader
    print(f"Train dataset size: {len(train_loader.dataset)}")

    # 检查训练集与测试集的交集
    train_paths = set(item[0] for item in train_loader.dataset.data)
    train_query_common = train_paths.intersection(query_paths)
    train_gallery_common = train_paths.intersection(gallery_paths)
    print(f"Number of common image paths between train and query: {len(train_query_common)}")
    print(f"Number of common image paths between train and gallery: {len(train_gallery_common)}")
    if len(train_query_common) > 0:
        print(f"Sample common paths (train vs query, first 5): {list(train_query_common)[:5]}")
    if len(train_gallery_common) > 0:
        print(f"Sample common paths (train vs gallery, first 5): {list(train_gallery_common)[:5]}")

    # 检查身份交集
    train_pids = set(item[2] for item in train_loader.dataset.data)
    query_pids = set(item[2] for item in query_loader.dataset.data)
    gallery_pids = set(item[2] for item in gallery_loader.dataset.data)
    train_query_pid_common = train_pids.intersection(query_pids)
    train_gallery_pid_common = train_pids.intersection(gallery_pids)
    print(f"Number of common PIDs between train and query: {len(train_query_pid_common)}")
    print(f"Number of common PIDs between train and gallery: {len(train_gallery_pid_common)}")
    if len(train_query_pid_common) > 0:
        print(f"Sample common PIDs (train vs query, first 5): {list(train_query_pid_common)[:5]}")
    if len(train_gallery_pid_common) > 0:
        print(f"Sample common PIDs (train vs gallery, first 5): {list(train_gallery_pid_common)[:5]}")

    # 模型
    num_classes = 781  # CUHK03的身份数量
    print(f"Number of classes (fixed): {num_classes}")
    model = T2IReIDModel(net_config=args)

    # 加载检查点
    checkpoint = load_checkpoint(args.resume)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint  # 假设检查点本身是state_dict

    # 打印 state_dict 和模型参数名，检查是否匹配
    print("State dict keys (first 5):", list(state_dict.keys())[:5])
    print("Model state dict keys (first 5):", list(model.state_dict().keys())[:5])

    # 过滤不匹配的权重
    model_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    missing_keys = [k for k in model_dict.keys() if k not in filtered_state_dict]
    print(f"Missing keys in state_dict: {missing_keys}")
    model_dict.update(filtered_state_dict)
    model.load_state_dict(model_dict, strict=False)  # 允许部分加载
    print("Loaded checkpoint with partial weight matching.")

    model = nn.DataParallel(model)
    model.eval()

    # 评估
    evaluator = Evaluator_t2i(model)
    print("Test:")
    metrics = evaluator.evaluate(query_loader, gallery_loader, query_loader.dataset.data, gallery_loader.dataset.data)

    # 保存测试结果
    with open(os.path.join(log_dir, f'{checkpoint_name}_test_results.txt'), 'w') as f:
        f.write(f"mAP: {metrics['mAP']:.4f}\n")
        f.write(f"Rank-1: {metrics['rank1']:.4f}\n")
        f.write(f"Rank-5: {metrics['rank5']:.4f}\n")
        f.write(f"Rank-10: {metrics['rank10']:.4f}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test T2I-ReID model")
    # 项目根目录定位
    ROOT_DIR = Path(__file__).parent.parent  # 从 scripts/ 到 TextGuidedReID/
    # Data
    parser.add_argument('--config', default=os.path.join(ROOT_DIR, 'configs', 'config_cuhk_pedes.yaml'))
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--width', type=int, default=224)
    parser.add_argument('--query-list', type=str, required=True)
    parser.add_argument('--gallery-list', type=str, required=True)
    parser.add_argument('--root', type=str, default=os.path.join(ROOT_DIR, 'data'))
    parser.add_argument('--vis_root', type=str, default=os.path.join(ROOT_DIR, 'data', 'vis', 't2i'))
    # Model
    parser.add_argument('--resume', type=str, required=True, metavar='PATH')
    # Testing
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    args.data_config = {'json_file': config['json_file']}
    for k, v in config.items():
        if k not in args.__dict__ or args.__dict__[k] == parser.get_default(k):
            setattr(args, k, v)

    main_worker(args)