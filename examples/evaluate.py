import argparse
import torch
from reid.models.pass_transformer_joint import T2IReIDModel
from reid.datasets.data_builder_t2i import DataBuilder_t2i
from reid.evaluation.evaluators_t import Evaluator_t2i


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate T2I-ReID model")
    parser.add_argument('--config', default='scripts/config_cuhk_pedes.yaml', help='Path to config file')
    parser.add_argument('--root', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--dataset-configs', nargs='+', type=str, required=True,
                        help='Dataset configurations in JSON format')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for evaluation')
    parser.add_argument('--workers', type=int, default=0, help='Number of data loading workers')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision evaluation')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 解析 dataset_configs
    args.dataset_configs = [eval(cfg) for cfg in args.dataset_configs]

    # 数据加载
    data_builder = DataBuilder_t2i(args, is_distributed=False)
    query_loader, gallery_loader = data_builder.build_data(is_train=False)

    # 模型加载
    model = T2IReIDModel(net_config=args)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    # 评估
    evaluator = Evaluator_t2i(model)
    with torch.no_grad():
        metrics = evaluator.evaluate(query_loader, gallery_loader, query_loader.dataset.data,
                                     gallery_loader.dataset.data)
    print("Final Evaluation Metrics:", metrics)


if __name__ == '__main__':
    main()
