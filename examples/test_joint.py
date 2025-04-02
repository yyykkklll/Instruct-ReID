import argparse
import os.path as osp
import sys

# 添加项目根目录到 sys.path
project_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch.nn as nn
import yaml

from reid.datasets.data_builder_t2i import DataBuilder_t2i
from reid.evaluation.evaluators_t import Evaluator_t2i
from reid.models.pass_transformer_joint import T2IReIDModel
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, copy_state_dict


def main_worker(args):
    log_dir = osp.dirname(args.resume)
    checkpoint_name = osp.basename(args.resume).split('.')[0]
    sys.stdout = Logger(osp.join(log_dir, f'{checkpoint_name}_log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # 数据加载
    data_builder = DataBuilder_t2i(args)
    query_loader, gallery_loader = data_builder.build_data(is_train=False)

    # 模型
    # 固定 num_classes 为训练时的值
    num_classes = 781  # 训练时的身份数量
    print(f"Number of classes (fixed): {num_classes}")
    model = T2IReIDModel(num_classes=num_classes, net_config=args)

    # 加载检查点
    checkpoint = load_checkpoint(args.resume)
    # 提取 state_dict
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint  # 假设 checkpoint 本身是 state_dict

    # 打印 state_dict 和模型参数名，检查是否匹配
    print("State dict keys (first 5):", list(state_dict.keys())[:5])
    print("Model state dict keys (first 5):", list(model.state_dict().keys())[:5])

    copy_state_dict(state_dict, model, strip='module.')

    model = nn.DataParallel(model)
    model.eval()

    # 评估
    evaluator = Evaluator_t2i(model)
    print("Test:")
    metrics = evaluator.evaluate(query_loader, gallery_loader, query_loader.dataset.data, gallery_loader.dataset.data)

    # 保存测试结果
    with open(osp.join(log_dir, f'{checkpoint_name}_test_results.txt'), 'w') as f:
        f.write(f"mAP: {metrics['mAP']:.4f}\n")
        f.write(f"Rank-1: {metrics['rank1']:.4f}\n")
        f.write(f"Rank-5: {metrics['rank5']:.4f}\n")
        f.write(f"Rank-10: {metrics['rank10']:.4f}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test T2I-ReID model")
    # Data
    parser.add_argument('--config', default='scripts/config_cuhk_pedes.yaml')
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--width', type=int, default=224)
    parser.add_argument('--query-list', type=str, required=True)
    parser.add_argument('--gallery-list', type=str, required=True)
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--vis_root', type=str, default='data/vis/t2i')
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
