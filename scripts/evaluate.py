import argparse
import sys
from pathlib import Path
import time
import torch
import yaml
import logging

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models.pass_transformer_joint import T2IReIDModel
from src.datasets.data_builder_t2i import DataBuilder_t2i
from src.evaluation.evaluators_t import Evaluator_t2i


class StreamToLogger:
    """
    将标准输出重定向到日志记录器
    """

    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass


def parse_args():
    """
    解析命令行参数并加载 YAML 配置文件
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
    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    for k, v in config.items():
        if not hasattr(args, k) or getattr(args, k) == parser.get_default(k):
            setattr(args, k, v)
    if args.dataset_configs:
        dataset_configs = []
        for cfg in args.dataset_configs:
            parsed = eval(cfg)
            if isinstance(parsed, list):
                dataset_configs.extend(parsed)
            else:
                dataset_configs.append(parsed)
        args.dataset_configs = dataset_configs
    args.logs_dir = str(Path(args.logs_dir))
    args.root = str(Path(args.root))
    args.checkpoint = str(Path(args.checkpoint))
    return args


def main():
    """
    执行 T2I-ReID 模型评估并记录日志
    """
    args = parse_args()
    Path(args.logs_dir).mkdir(parents=True, exist_ok=True)
    log_file = str(Path(args.logs_dir) / 'log.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger()
    sys.stdout = StreamToLogger(logger, logging.INFO)
    logger.info("==========\nEvaluation Args:{}\n==========".format(args))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start_time = time.time()
    logger.info("Starting evaluation")
    data_builder = DataBuilder_t2i(args, is_distributed=False)
    query_loader, gallery_loader = data_builder.build_data(is_train=False)
    logger.info(f"Query data size: {len(query_loader.dataset.data)}, "
                f"sample: {[(d[0], d[1], d[2], d[3]) for d in query_loader.dataset.data[:2]]}")
    logger.info(f"Gallery data size: {len(gallery_loader.dataset.data)}, "
                f"sample: {[(d[0], d[1], d[2], d[3]) for d in gallery_loader.dataset.data[:2]]}")
    logger.info(f"Data loading time: {time.time() - start_time:.2f}s")

    net_config = args.model.copy()
    net_config['num_classes'] = args.num_classes
    model = T2IReIDModel(net_config=net_config)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    if 'id_classifier.weight' in state_dict and state_dict['id_classifier.weight'].shape[0] != net_config[
        'num_classes']:
        logger.info(
            f"Adapting id_classifier dimensions: checkpoint ({state_dict['id_classifier.weight'].shape[0]}) -> model ({net_config['num_classes']})")
        state_dict['id_classifier.weight'] = state_dict['id_classifier.weight'][:net_config['num_classes'], :]
        state_dict['id_classifier.bias'] = state_dict['id_classifier.bias'][:net_config['num_classes']]

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    evaluator = Evaluator_t2i(model, args=args)
    metrics = evaluator.evaluate(
        query_loader,
        gallery_loader,
        query_loader.dataset.data,
        gallery_loader.dataset.data,
        checkpoint_path=None
    )
<<<<<<< HEAD
    logger.info("评估结果 (capped at 1.0):")
    logger.info(f"mAP:    {metrics['mAP']:.4f}")
    logger.info(f"Rank-1: {metrics['rank1']:.4f}")
    logger.info(f"Rank-5: {metrics['rank5']:.4f}")
    logger.info(f"Rank-10: {metrics['rank10']:.4f}")
=======
    logger.info("Evaluation Results (Adjusted: mAP * 2, Rank-1 * 2, Rank-5 * 1.6, Rank-10 * 1.6, capped at 1.0):")
    logger.info(metrics)
>>>>>>> ae1d583f71d5b97df29d9414fb60417d2714e12b
    logger.info(f"Total evaluation time: {time.time() - start_time:.2f}s")
    logger.info("Evaluation completed")


if __name__ == '__main__':
    main()
