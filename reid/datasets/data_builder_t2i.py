import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np


def merge_sub_datasets(train_list, train_root_list, args):
    if not isinstance(train_list, list):
        task_list = [train_list]
    else:
        task_list = train_list
    if not isinstance(train_root_list, list):
        task_pref = [train_root_list]
    else:
        task_pref = train_root_list

    assert len(task_list) == len(task_pref), "Number of datasets and roots must match"

    list_lines_all = []
    global_pid_list = {}  # 用于映射 pid 到 [0, num_classes-1]
    global_pid_counter = 0

    # 加载文本描述
    json_file = args.data_config.get('json_file', '')
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"JSON file not found at: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        attr_dict = json.load(f)

    for list_file, prefix in zip(task_list, task_pref):
        with open(list_file) as f:
            lines = f.readlines()
            for line in lines:
                info = line.strip('\n').split(" ")
                # 新格式：img_path pid cam_id
                imgs = info[0]
                pids = info[1]
                cam_id = info[2]
                # 映射 pid 到 [0, num_classes-1]
                if pids not in global_pid_list:
                    global_pid_list[pids] = global_pid_counter
                    global_pid_counter += 1
                mapped_pid = global_pid_list[pids]
                view_id = "0"  # 默认值，CUHK-PEDES 不使用 view_id
                # 从 attr_dict 获取真实文本描述
                # 移除后缀（如果有）
                img_key = imgs
                if img_key.endswith('_0') or img_key.endswith('_1'):
                    img_key = img_key.rsplit('_', 1)[0]
                entry = attr_dict.get(img_key, {"caption": f"Person with ID {pids}", "id": pids})
                caption = entry["caption"]
                # 确保路径格式正确
                imgs = imgs.replace('\\', '/')
                if not imgs.endswith(('.png', '.jpg', '.jpeg')):
                    imgs += '.png'
                full_path = os.path.join(prefix, imgs).replace('\\', '/')
                list_lines_all.append((full_path, caption, mapped_pid, view_id, cam_id))

    # 保存 global_pid_list 供后续使用（例如测试时）
    args.global_pid_list = global_pid_list
    return list_lines_all


class DataBuilder_t2i:
    def __init__(self, args, is_distributed=False):
        self.args = args
        self.root = args.root
        self.train_list = getattr(args, 'train_list', None)
        self.query_list = args.query_list
        self.gallery_list = args.gallery_list
        self.is_distributed = is_distributed
        self.json_file = args.data_config.get('json_file', '')

    def _load_data(self, list_lines):
        data = []
        for img_path, caption, person_id, view_id, cam_id in list_lines:
            person_id = int(person_id)  # 已经是映射后的值
            cam_id = int(cam_id)
            data.append((img_path, caption, person_id, cam_id))
        return data

    def build_dataset(self, data, is_train=False):
        class CUHKPEDESDataset(Dataset):
            def __init__(self, data, args, transform=None):
                self.data = data
                self.args = args
                self.transform = transform

            def __len__(self):
                return len(self.data)

            def __getitem__(self, index):
                img_path, caption, pid, cam_id = self.data[index]
                img_path = img_path.replace('/', os.sep).replace('\\', os.sep)

                try:
                    image = Image.open(img_path).convert('RGB')
                except Exception as e:
                    print(f"Warning: Failed to load image {img_path}, using zero image: {e}")
                    image_array = np.zeros((self.args.height, self.args.width, 3), dtype=np.uint8)
                    image = Image.fromarray(image_array)

                if self.transform is not None:
                    image = self.transform(image)

                pid_tensor = torch.tensor(pid, dtype=torch.long)
                cam_id_tensor = torch.tensor(cam_id, dtype=torch.long)

                return image, caption, pid_tensor, cam_id_tensor

        # 定义训练和测试的 transform
        if is_train:
            transform = transforms.Compose([
                transforms.Resize((self.args.height + 32, self.args.width + 32)),  # 稍微放大以便随机裁剪
                transforms.RandomCrop((self.args.height, self.args.width), padding=4),  # 随机裁剪
                transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
                transforms.RandomRotation(degrees=10),  # 随机旋转，幅度较小
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((self.args.height, self.args.width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        dataset = CUHKPEDESDataset(data, self.args, transform=transform)
        return dataset

    def _build_train_loader(self, data):
        dataset = self.build_dataset(data, is_train=True)

        def collate_fn(batch):
            images, captions, pids, cam_ids = zip(*batch)
            images = torch.stack(images, dim=0)
            pids = torch.stack(pids, dim=0)
            cam_ids = torch.stack(cam_ids, dim=0)
            return images, captions, pids, cam_ids

        train_loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn
        )
        return train_loader, dataset

    def _build_test_loader(self, data, is_query=False):
        dataset = self.build_dataset(data, is_train=False)

        def collate_fn(batch):
            images, captions, pids, cam_ids = zip(*batch)
            images = torch.stack(images, dim=0)
            pids = torch.stack(pids, dim=0)
            cam_ids = torch.stack(cam_ids, dim=0)
            return images, captions, pids, cam_ids

        test_loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn
        )
        return test_loader

    def build_data(self, is_train=False):
        if is_train:
            list_lines = merge_sub_datasets(self.train_list, self.root, self.args)
            data = self._load_data(list_lines)
            print("Dataset statistics:")
            print("  ------------------------------------------")
            print("  subset   | # ids | # images | # cameras")
            print("  ------------------------------------------")
            print("  train    | {:5d} | {:8d}   | {:8d}".format(
                len(set([item[2] for item in data])),
                len(data),
                len(set([item[3] for item in data]))
            ))
            print("  ------------------------------------------")
            return self._build_train_loader(data)
        else:
            list_lines = merge_sub_datasets(self.query_list, self.root, self.args)
            query_data = self._load_data(list_lines)
            list_lines = merge_sub_datasets(self.gallery_list, self.root, self.args)
            gallery_data = self._load_data(list_lines)
            query_loader = self._build_test_loader(query_data, is_query=True)
            gallery_loader = self._build_test_loader(gallery_data, is_query=False)
            return query_loader, gallery_loader