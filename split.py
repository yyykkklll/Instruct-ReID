import json
import os

# 读取 annotations.json
with open('cuhk_pedes/annotations/annotations.json', 'r') as f:
    annotations = json.load(f)

# 提取所有身份
pid_to_images = {}
for img_path, info in annotations.items():
    pid = info['id']
    if pid not in pid_to_images:
        pid_to_images[pid] = []
    pid_to_images[pid].append(img_path)

# 统计身份数量
num_pids = len(pid_to_images)
print(f"Total number of PIDs: {num_pids}")  # 应为 843

# 划分身份（调整为从 1 开始的 PID）
train_pids = list(range(1, 613))  # PID: 1-612
test_pids = list(range(613, 782))  # PID: 613-781
print(f"Train PIDs: {len(train_pids)}, Test PIDs: {len(test_pids)}")

# 生成训练集
train_lines = []
for pid in train_pids:
    if pid in pid_to_images:  # 添加检查
        for img_path in pid_to_images[pid]:
            train_lines.append(f"{img_path} {pid} 0\n")  # CamID 设为 0

# 生成测试集（查询和图库）
query_lines = []
gallery_lines = []
for pid in test_pids:
    images = pid_to_images[pid]
    # 按 1:1 划分查询和图库（如果图像数量不足，查询和图库可能有重叠）
    num_images = len(images)
    split_idx = max(1, num_images // 2)  # 至少 1 张图像
    query_images = images[:split_idx]
    gallery_images = images[split_idx:] if num_images > 1 else images
    for img_path in query_images:
        query_lines.append(f"{img_path} {pid} 0\n")
    for img_path in gallery_images:
        gallery_lines.append(f"{img_path} {pid} 0\n")

# 添加干扰样本到图库
distractor_pids = train_pids[:50]  # 从训练集中选择 50 个身份作为干扰样本
for pid in distractor_pids:
    images = pid_to_images[pid]
    # 每个身份随机选择 5 张图像（如果不足 5 张，则全部选择）
    num_distractors = min(5, len(images))
    distractor_images = images[:num_distractors]
    for img_path in distractor_images:
        gallery_lines.append(f"{img_path} {pid} 0\n")

# 保存文件
with open('cuhk_pedes/splits/train_t2i_new.txt', 'w') as f:
    f.writelines(train_lines)
with open('cuhk_pedes/splits/query_t2i_v2_new.txt', 'w') as f:
    f.writelines(query_lines)
with open('cuhk_pedes/splits/gallery_t2i_v2_new.txt', 'w') as f:
    f.writelines(gallery_lines)

# 打印统计信息
print(f"New train dataset size: {len(train_lines)} images")
print(f"New query dataset size: {len(query_lines)} images")
print(f"New gallery dataset size: {len(gallery_lines)} images")