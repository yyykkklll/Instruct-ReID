import argparse
import sys
from pathlib import Path
import time
import torch
import yaml
import logging
import json

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models.pass_transformer_joint import T2IReIDModel
from src.datasets.data_builder_t2i import DataBuilder_t2i
from src.evaluation.evaluators_t import Evaluator_t2i


def parse_args():
    """
    解析命令行参数并加载 YAML 配置文件。

    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(description="Evaluate T2I-ReID model")
    parser.add_argument('--config', default=str(ROOT_DIR / 'configs' / 'config_cuhk_pedes.yaml'),
                        help='Path to config file')
    parser.add_argument('--root', type=str, default=str(ROOT_DIR / 'data'),
                        help='Root directory of the dataset')
    parser.add_argument('--dataset-configs', nargs='+', type=str, required=True,
                        help='Dataset configurations in JSON format')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for evaluation')
    parser.add_argument('--workers', type=int, default=0, help='Number of data loading workers')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision evaluation')
    parser.add_argument('--logs-dir', type=str, default=str(ROOT_DIR / 'logs'), help='Directory for logs')
    args = parser.parse_args()

    # 加载 YAML 配置文件
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with config_path.open('r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 覆盖默认参数
    for k, v in config.items():
        if not hasattr(args, k) or getattr(args, k) == parser.get_default(k):
            setattr(args, k, v)

    # 解析数据集配置（优化：使用 json.loads 替代 eval）
    dataset_configs = []
    for cfg in args.dataset_configs:
        try:
            parsed = json.loads(cfg)
            dataset_configs.extend(parsed if isinstance(parsed, list) else [parsed])
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse dataset config: {cfg}, error: {e}")
            raise
    args.dataset_configs = dataset_configs

    # 统一路径并验证（优化：合并检查）
    paths_to_check = {
        'logs_dir': Path(args.logs_dir),
        'root': Path(args.root),
        'checkpoint': Path(args.checkpoint)
    }
    for name, path in paths_to_check.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} not found at: {path}")
        setattr(args, name, str(path))

    return args


def main():
    """
    执行 T2I-ReID 模型评估并记录日志。
    """
    args = parse_args()
    Path(args.logs_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(Path(args.logs_dir) / 'log.txt', mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Evaluation Args: {vars(args)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start_time = time.time()
    logging.info("Starting evaluation")

    # 构建数据集
    data_builder = DataBuilder_t2i(args, is_distributed=False)
    query_loader, gallery_loader = data_builder.build_data(is_train=False)
    logging.info(f"Dataset loaded: Query={len(query_loader.dataset.data)}, "
                 f"Gallery={len(gallery_loader.dataset.data)}, Time={time.time() - start_time:.2f}s")

    # 初始化模型
    model = T2IReIDModel(net_config=args.model)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))

    # 调整分类器维度（优化：简化日志）
    if 'id_classifier.weight' in state_dict and state_dict['id_classifier.weight'].shape[0] != args.model['num_classes']:
        logging.info(f"Adapting id_classifier: {state_dict['id_classifier.weight'].shape[0]} -> {args.model['num_classes']}")
        state_dict['id_classifier.weight'] = state_dict['id_classifier.weight'][:args.model['num_classes'], :]
        state_dict['id_classifier.bias'] = state_dict['id_classifier.bias'][:args.model['num_classes']]

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    # 评估（优化：使用 torch.no_grad）
    with torch.no_grad():
        evaluator = Evaluator_t2i(model, args=args)
        metrics = evaluator.evaluate(
            query_loader, gallery_loader,
            query_loader.dataset.data, gallery_loader.dataset.data,
            checkpoint_path=None
        )

    # 记录指标
    logging.info("Evaluation Results:")
    logging.info(f"mAP:    {metrics['mAP']:.4f} ({metrics['mAP']*100:.2f}%)")
    logging.info(f"Rank-1: {metrics['rank1']:.4f} ({metrics['rank1']*100:.2f}%)")
    logging.info(f"Rank-5: {metrics['rank5']:.4f} ({metrics['rank5']*100:.2f}%)")
    logging.info(f"Rank-10: {metrics['rank10']:.4f} ({metrics['rank10']*100:.2f}%)")
    logging.info(f"Total evaluation time: {time.time() - start_time:.2f}s")
    logging.info("Evaluation completed!")


if __name__ == '__main__':
    main()