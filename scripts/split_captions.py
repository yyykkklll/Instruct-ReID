import argparse
import json
import re
from pathlib import Path


def split_captions(input_json, output_cloth_json, output_id_json, dataset_name):
    """
    分离数据集的文本描述为衣物描述和身份描述，生成新的 JSON 文件。

    Args:
        input_json (Path): 输入标注文件路径
        output_cloth_json (Path): 输出的衣物描述文件路径
        output_id_json (Path): 输出的身份描述文件路径
        dataset_name (str): 数据集名称，用于适配不同格式
    """
    # 读取输入 JSON
    with input_json.open('r') as f:
        data = json.load(f)

    # 定义关键词
    cloth_keywords = r'\b(wear|wears|wearing|dress|shirt|pants|jacket|coat|skirt|top|bottom|sweater|color|pattern|sleeve|length|shoe|shoes|hat|cap|scarf|belt|collar|neck|floral|patterned)\b'
    id_keywords = r'\b(man|woman|male|female|person|heavy|slim|tall|short|age|build|set|body|height|gender|face|hair|young|old|middle-aged|curly|straight|length|shoulder-length|neck-length)\b'

    cloth_captions = []
    id_captions = []

    # 处理数据集
    for item in data:
        captions = item.get('captions', [])
        if not isinstance(captions, list):
            captions = [captions]  # 转换为列表

        cloth_sentences = []
        id_sentences = []

        # 遍历每个 caption
        for caption in captions:
            if not caption.strip():
                continue
            # 按句子分割
            sentences = [s.strip() for s in caption.split('.') if s.strip()]
            for sentence in sentences:
                # 优先检查衣物关键词
                if re.search(cloth_keywords, sentence, re.IGNORECASE):
                    cloth_sentences.append(sentence)
                # 再检查身份关键词（避免重复添加）
                elif re.search(id_keywords, sentence, re.IGNORECASE):
                    id_sentences.append(sentence)

        # 创建新条目，保留原始字段
        cloth_item = item.copy()
        id_item = item.copy()
        cloth_item['captions'] = cloth_sentences if cloth_sentences else []
        id_item['captions'] = id_sentences if id_sentences else []

        cloth_captions.append(cloth_item)
        id_captions.append(id_item)

    # 保存输出 JSON
    output_cloth_json.parent.mkdir(parents=True, exist_ok=True)
    output_id_json.parent.mkdir(parents=True, exist_ok=True)

    with output_cloth_json.open('w') as f:
        json.dump(cloth_captions, f, indent=4)
    with output_id_json.open('w') as f:
        json.dump(id_captions, f, indent=4)

    print(f"Generated {output_cloth_json} with {len(cloth_captions)} entries")
    print(f"Generated {output_id_json} with {len(id_captions)} entries")


def main():
    parser = argparse.ArgumentParser(description="Split dataset captions into cloth and ID descriptions")
    parser.add_argument('--dataset', type=str, required=True, choices=['CUHK-PEDES', 'ICFG-PEDES', 'RSTPReid'],
                        help="Dataset name")
    args = parser.parse_args()

    # 数据集配置
    dataset_configs = {
        'CUHK-PEDES': {
            'input_json': Path('data/CUHK-PEDES/annotations/caption_all.json'),
            'output_cloth_json': Path('data/CUHK-PEDES/annotations/caption_cloth.json'),
            'output_id_json': Path('data/CUHK-PEDES/annotations/caption_id.json')
        },
        'ICFG-PEDES': {
            'input_json': Path('data/ICFG-PEDES/annotations/ICFG-PEDES.json'),
            'output_cloth_json': Path('data/ICFG-PEDES/annotations/caption_cloth.json'),
            'output_id_json': Path('data/ICFG-PEDES/annotations/caption_id.json')
        },
        'RSTPReid': {
            'input_json': Path('data/RSTPReid/annotations/data_captions.json'),
            'output_cloth_json': Path('data/RSTPReid/annotations/caption_cloth.json'),
            'output_id_json': Path('data/RSTPReid/annotations/caption_id.json')
        }
    }

    config = dataset_configs[args.dataset]
    split_captions(
        input_json=config['input_json'],
        output_cloth_json=config['output_cloth_json'],
        output_id_json=config['output_id_json'],
        dataset_name=args.dataset
    )


if __name__ == '__main__':
    main()
