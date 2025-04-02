import json
import os
from tqdm import tqdm  # 添加 tqdm 导入

# 使用绝对路径并创建必要的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
annotations_dir = os.path.join(current_dir, 'cuhk_pedes', 'annotations')
os.makedirs(annotations_dir, exist_ok=True)

# 读取 reid_raw.json
reid_raw_path = os.path.join(current_dir, 'reid_raw.json')
with open(reid_raw_path, 'r') as f:
    raw_data = json.load(f)

# 转换后的 annotations.json
annotations = {}

# 假设 CUHK03 图像目录
image_dir = 'cuhk_pedes/images/CUHK03'
image_files = set()
for root, _, files in os.walk(image_dir):
    for file in files:
        if file.endswith('.png'):
            rel_path = os.path.join('CUHK03', file).replace('\\', '/')
            image_files.add(rel_path)

# 遍历 reid_raw.json
for entry in tqdm(raw_data, desc="处理图像匹配"):
    pid = entry['id']
    captions = entry['captions']
    # 为每个身份生成对应的 CUHK03 图像路径
    for img_file in image_files:
        # 提取 pid，例如 1_0001_1_01.png 中的 0001
        img_pid = int(img_file.split('_')[1])
        if img_pid == pid:
            caption = captions[0] if captions else f"Person with ID {pid}"
            annotations[img_file] = {
                "caption": caption,
                "id": pid
            }

# 保存为 annotations.json（使用绝对路径）
output_path = os.path.join(annotations_dir, 'annotations.json')
with open(output_path, 'w') as f:
    json.dump(annotations, f, indent=4)