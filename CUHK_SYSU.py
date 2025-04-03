import scipy.io as sio
import json
import os
import numpy as np

# 路径设置
dataset_root = "CUHK-SYSU"
image_dir = os.path.join(dataset_root, "Image/SSM")
annotation_dir = os.path.join(dataset_root, "annotation")
output_dir = os.path.join(dataset_root, "processed")
os.makedirs(output_dir, exist_ok=True)

# 加载 .mat 文件
images_mat_path = os.path.join(annotation_dir, "Images.mat")
print(f"Loading Images.mat from: {images_mat_path}")
images_mat = sio.loadmat(images_mat_path)
print(f"Images.mat 中的键名： {list(images_mat.keys())}")
images_mat = images_mat["Img"][0]

person_mat = sio.loadmat(os.path.join(annotation_dir, "Person.mat"))["Person"][0]
train_mat = sio.loadmat(os.path.join(annotation_dir, "test/train_test/Train.mat"))["Train"][0]
test_mat = sio.loadmat(os.path.join(annotation_dir, "test/train_test/TestG100.mat"))["TestG100"][0]

# 构建 annotations.json
annotations = {}
for img_info in images_mat:
    img_name = img_info["imname"][0]
    img_path = os.path.join("Image/SSM", img_name).replace("\\", "/")
    caption = f"Person in image {img_name}"
    annotations[img_path] = {"caption": caption, "id": -1}

# 创建身份映射字典
pid_mapping = {}
current_pid = 0

# 从 Person.mat 中提取身份信息
person_to_images = {}
for person_info in person_mat:
    person_id_str = str(person_info["idname"][0])
    if person_id_str not in pid_mapping:
        pid_mapping[person_id_str] = current_pid
        current_pid += 1
    person_id = pid_mapping[person_id_str]

    scenes = person_info["scene"][0]
    for scene in scenes:
        # 获取场景中的图像名称
        img_name = scene["imname"][0]
        img_path = os.path.join("Image/SSM", img_name).replace("\\", "/")
        
        if img_path not in annotations:
            print(f"Warning: img_path {img_path} not found in annotations.")
        else:
            annotations[img_path]["id"] = person_id
            if person_id not in person_to_images:
                person_to_images[person_id] = []
            person_to_images[person_id].append(img_path)

# 统计 id 分配情况
id_counts = {}
for img_path, info in annotations.items():
    id_val = info["id"]
    id_counts[id_val] = id_counts.get(id_val, 0) + 1
print(f"ID distribution: {id_counts}")

# 保存 annotations.json
with open(os.path.join(output_dir, "annotations.json"), "w") as f:
    json.dump(annotations, f)

# 生成 train_list（只使用 id >= 0 的图像）
train_pids = set()
train_lines = []
train_pid_mapping = {}
train_pid_counter = 0

for train_info in train_mat:
    person_id_str = str(train_info["idname"][0])
    if person_id_str not in pid_mapping:
        continue
    person_id = pid_mapping[person_id_str]
    if person_id_str not in train_pid_mapping:
        train_pid_mapping[person_id_str] = train_pid_counter
        train_pid_counter += 1
    mapped_pid = train_pid_mapping[person_id_str]
    train_pids.add(mapped_pid)
    if person_id in person_to_images:
        for img_path in person_to_images[person_id]:
            if annotations[img_path]["id"] >= 0:
                train_lines.append(f"{img_path} {mapped_pid} 0\n")

# 生成 query_list 和 gallery_list（只使用 id >= 0 的图像）
test_pids = set()
query_lines = []
gallery_lines = []
test_pid_mapping = {}
test_pid_counter = 0

print("正在处理测试集数据...")
for test_info in test_mat:
    # 检查测试集数据结构
    print(f"测试集数据字段: {test_info.dtype.names}")
    
    # 处理查询集（Query）
    if "Query" in test_info.dtype.names:
        query_info = test_info["Query"][0]
        if isinstance(query_info, np.ndarray):
            for q_info in query_info:
                if "idname" in q_info.dtype.names:
                    person_id_str = str(q_info["idname"][0])
                    if person_id_str not in pid_mapping:
                        continue
                    person_id = pid_mapping[person_id_str]
                    if person_id_str not in test_pid_mapping:
                        test_pid_mapping[person_id_str] = test_pid_counter
                        test_pid_counter += 1
                    mapped_pid = test_pid_mapping[person_id_str]
                    test_pids.add(mapped_pid)
                    if person_id in person_to_images:
                        for img_path in person_to_images[person_id]:
                            if annotations[img_path]["id"] >= 0:
                                query_lines.append(f"{img_path} {mapped_pid} 0\n")
    
    # 处理图库集（Gallery）
    if "Gallery" in test_info.dtype.names:
        gallery_info = test_info["Gallery"][0]
        if isinstance(gallery_info, np.ndarray):
            for g_info in gallery_info:
                if "imname" in g_info.dtype.names:
                    img_name = g_info["imname"][0]
                    img_path = os.path.join("Image/SSM", img_name).replace("\\", "/")
                    if img_path in annotations and annotations[img_path]["id"] >= 0:
                        gallery_lines.append(f"{img_path} {annotations[img_path]['id']} 0\n")

# 添加干扰样本到图库
distractor_pids = list(train_pids)[:50]
for pid in distractor_pids:
    for pid_str, mapped_pid in train_pid_mapping.items():
        if mapped_pid == pid:
            person_id = pid_mapping[pid_str]
            break
    images = person_to_images.get(person_id, [])
    num_distractors = min(5, len(images))
    distractor_images = images[:num_distractors]
    for img_path in distractor_images:
        if annotations[img_path]["id"] >= 0:
            gallery_lines.append(f"{img_path} {pid} 0\n")

# 保存列表文件
with open(os.path.join(output_dir, "train_list.txt"), "w") as f:
    f.writelines(train_lines)
with open(os.path.join(output_dir, "query_list.txt"), "w") as f:
    f.writelines(query_lines)
with open(os.path.join(output_dir, "gallery_list.txt"), "w") as f:
    f.writelines(gallery_lines)

# 打印统计信息
print(f"Train dataset size: {len(train_lines)} images, {len(train_pids)} identities")
print(f"Query dataset size: {len(query_lines)} images, {len(test_pids)} identities")
print(f"Gallery dataset size: {len(gallery_lines)} images")