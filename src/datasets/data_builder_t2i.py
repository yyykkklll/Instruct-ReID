import json
import os
from pathlib import Path
import random

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader


def merge_sub_datasets(dataset_configs, args):
    list_lines_all = []
    global_pid_list = {}
    global_pid_counter = 0

    ROOT_DIR = Path(__file__).parent.parent.parent  # 从 src/datasets/ 到 v3/

    for config in dataset_configs:
        dataset_name = config['name']
        prefix = os.path.join(ROOT_DIR, config.get('root', ''))
        json_file_raw = config.get('json_file', '')
        json_file = os.path.join(ROOT_DIR, json_file_raw)

        if not os.path.exists(json_file):
            raise FileNotFoundError(f"JSON file not found for {dataset_name} at: {json_file}")
        with open(json_file, 'r', encoding='utf-8') as f:
            attr_dict_raw = json.load(f)

        # 按 ID 分组
        items_by_id = {}
        for item in attr_dict_raw:
            img_path = item.get('file_path', item.get('img_path', ''))
            pid = str(item['id'])
            caption = item['captions'][0] if 'captions' in item else ' '.join(item['processed_tokens'][0])
            ext = '.jpg'  # 默认扩展名，动态调整

            if pid not in global_pid_list:
                global_pid_list[pid] = global_pid_counter
                global_pid_counter += 1
            mapped_pid = global_pid_list[pid]

            cam_id = "0"
            full_path = os.path.join(prefix, img_path).replace('\\', os.sep)
            if not full_path.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                full_path += ext

            if not os.path.exists(full_path):
                print(f"Warning: Image not found at {full_path}")

            if pid not in items_by_id:
                items_by_id[pid] = []
            items_by_id[pid].append((full_path, caption, mapped_pid, cam_id))

        # 划分数据集
        all_ids = list(items_by_id.keys())
        random.seed(0)  # 固定种子
        random.shuffle(all_ids)

        train_ratio = 0.85
        train_id_count = int(len(all_ids) * train_ratio)
        train_ids = all_ids[:train_id_count]
        test_ids = all_ids[train_id_count:]

        # 测试集：选择目标 ID 数
        target_test_ids = test_ids[:min(1000 if dataset_name == "CUHK-PEDES" else 608 if dataset_name == "ICFG-PEDES" else 615, len(test_ids))]

        train_lines = []
        query_lines = []
        gallery_lines = []

        # 训练集
        for pid in train_ids:
            train_lines.extend(items_by_id[pid])

        # 测试集：查询和图库
        for pid in target_test_ids:
            items = items_by_id[pid]
            random.shuffle(items)
            split_point = max(1, len(items) // 2)  # 平均分配，至少 1 张
            query_lines.extend(items[:split_point])
            gallery_lines.extend(items[split_point:])

        if hasattr(args, 'is_train') and args.is_train:
            list_lines_all.extend(train_lines)
            print(f"{dataset_name} - Training set: {len(train_lines)} images, {len(set([x[2] for x in train_lines]))} IDs")
        else:
            list_lines_all.extend(query_lines + gallery_lines)
            args.query_data = query_lines
            args.gallery_data = gallery_lines
            print(f"{dataset_name} - Query set: {len(query_lines)} images, {len(set([x[2] for x in query_lines]))} IDs")
            print(f"{dataset_name} - Gallery set: {len(gallery_lines)} images, {len(set([x[2] for x in gallery_lines]))} IDs")
            common_ids = set([x[2] for x in query_lines]) & set([x[2] for x in gallery_lines])
            print(f"{dataset_name} - Common IDs between query and gallery: {len(common_ids)}")

    args.global_pid_list = global_pid_list
    return list_lines_all


class DataBuilder_t2i:
    def __init__(self, args, is_distributed=False):
        self.args = args
        self.is_distributed = is_distributed
        self.dataset_configs = args.dataset_configs

    def _load_data(self, list_lines):
        data = []
        for img_path, caption, person_id, cam_id in list_lines:
            data.append((img_path, caption, int(person_id), int(cam_id)))
        return data

    def build_dataset(self, data, is_train=False):
        class T2IReIDDataset(Dataset):
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
                    print(f"Warning: Failed to load image {img_path}: {e}")
                    image_array = torch.zeros(3, self.args.height, self.args.width)
                    image = Image.fromarray(image_array.numpy(), mode='RGB')

                if self.transform is not None:
                    image = self.transform(image)

                return image, caption, torch.tensor(pid, dtype=torch.long), torch.tensor(cam_id, dtype=torch.long)

        if is_train:
            transform = transforms.Compose([
                transforms.Resize((self.args.height, self.args.width)),
                transforms.RandomCrop((self.args.height, self.args.width), padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((self.args.height, self.args.width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        return T2IReIDDataset(data, self.args, transform=transform)

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
        self.args.is_train = is_train
        if is_train:
            list_lines = merge_sub_datasets(self.dataset_configs, self.args)
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
            list_lines = merge_sub_datasets(self.dataset_configs, self.args)
            query_data = self._load_data(self.args.query_data)
            gallery_data = self._load_data(self.args.gallery_data)
            query_loader = self._build_test_loader(query_data, is_query=True)
            gallery_loader = self._build_test_loader(gallery_data, is_query=False)
            return query_loader, gallery_loader