import json
import os
import random
from collections import defaultdict

# 设置随机种子以确保可重复性
random.seed(42)

# 路径设置
image_dir = 'cuhk_pedes/images/CUHK03'
annotations_file = 'cuhk_pedes/annotations/annotations.json'
train_file = 'cuhk_pedes/splits/train_t2i_v2.txt'
query_file = 'cuhk_pedes/splits/query_t2i_v2.txt'
gallery_file = 'cuhk_pedes/splits/gallery_t2i_v2.txt'

# 读取 annotations.json
with open(annotations_file, 'r') as f:
    annotations = json.load(f)

# 按身份组织图像
pid_to_images = defaultdict(list)
for img_path, info in annotations.items():
    pid = info['id']
    caption = info['caption']
    pid_to_images[pid].append((img_path, caption))

# 获取所有身份
all_pids = sorted(list(pid_to_images.keys()))
num_pids = len(all_pids)
print(f"Total number of identities: {num_pids}")

# 划分训练和测试身份（80% 训练，20% 测试）
train_ratio = 0.8
num_train_pids = int(num_pids * train_ratio)
train_pids = all_pids[:num_train_pids]
test_pids = all_pids[num_train_pids:]
print(f"Training identities: {len(train_pids)}, Test identities: {len(test_pids)}")

# 生成训练集
train_lines = []
for pid in train_pids:
    for img_path, caption in pid_to_images[pid]:
        # 格式：img_path pid cam_id
        # cam_id 设为 0（CUHK03 不使用摄像头信息）
        line = f"{img_path} {pid} 0\n"
        train_lines.append(line)

# 生成测试集（查询和图库）
query_lines = []
gallery_lines = []
for pid in test_pids:
    images = pid_to_images[pid]
    random.shuffle(images)  # 随机打乱图像
    num_images = len(images)
    if num_images < 2:
        # 如果该身份只有一张图像，放入图库
        img_path, caption = images[0]
        gallery_lines.append(f"{img_path} {pid} 0\n")
        continue
    # 至少两张图像，分为查询和图库
    split_idx = max(1, num_images // 2)  # 至少一张用于查询
    query_images = images[:split_idx]
    gallery_images = images[split_idx:]
    
    # 查询集：img_path pid cam_id
    for img_path, caption in query_images:
        line = f"{img_path} {pid} 0\n"
        query_lines.append(line)
    
    # 图库集：img_path pid cam_id
    for img_path, _ in gallery_images:
        line = f"{img_path} {pid} 0\n"
        gallery_lines.append(line)

# 保存文件
with open(train_file, 'w') as f:
    f.writelines(train_lines)
with open(query_file, 'w') as f:
    f.writelines(query_lines)
with open(gallery_file, 'w') as f:
    f.writelines(gallery_lines)

# 统计信息
print(f"Training set: {len(train_lines)} images")
print(f"Query set: {len(query_lines)} images")
print(f"Gallery set: {len(gallery_lines)} images")

# 验证身份覆盖
query_pids = set(line.split()[1] for line in query_lines)  # 调整索引，因为移除了 clothes 字段
gallery_pids = set(line.split()[1] for line in gallery_lines)
print(f"Query PIDs: {len(query_pids)}, Gallery PIDs: {len(gallery_pids)}, Intersection: {len(query_pids & gallery_pids)}")