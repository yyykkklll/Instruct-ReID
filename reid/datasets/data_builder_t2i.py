import json
import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader


def merge_sub_datasets(dataset_configs, args):
    """
    合并多个数据集（如CUHK-PEDES、ICFG-PEDES、RSTPReid），生成统一的数据列表。
    dataset_configs: 列表，包含每个数据集的配置（名称、根目录、标注文件）。
    """
    list_lines_all = []
    global_pid_list = {}  # 映射原始 pid 到连续的 [0, num_classes-1]
    global_pid_counter = 0

    for config in dataset_configs:
        dataset_name = config['name']
        prefix = config['root']
        json_file = config.get('json_file', '')

        # 加载文本描述
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"JSON file not found for {dataset_name} at: {json_file}")
        with open(json_file, 'r', encoding='utf-8') as f:
            attr_dict_raw = json.load(f)

        # 从 JSON 中提取数据
        train_lines = []
        query_lines = []
        gallery_lines = []

        for item in attr_dict_raw:
            # 根据数据集类型获取图像路径和描述
            if dataset_name == "CUHK-PEDES":
                img_path = item['file_path']
                pid = str(item['id'])
                caption = item['captions'][0]  # 选择第一个 caption
            elif dataset_name == "ICFG-PEDES":
                img_path = item['file_path']
                pid = str(item['id'])
                caption = item['captions'][0]  # 只有一个 caption
            elif dataset_name == "RSTPReid":
                img_path = item['img_path']
                pid = str(item['id'])
                caption = item['captions'][0]  # 选择第一个 caption
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            # 映射 pid 到连续值
            if pid not in global_pid_list:
                global_pid_list[pid] = global_pid_counter
                global_pid_counter += 1
            mapped_pid = global_pid_list[pid]

            # 默认 cam_id（文件树中无明确相机信息）
            cam_id = "0"

            # 规范化图像路径
            full_path = os.path.join(prefix, img_path).replace('\\', '/')
            if not full_path.endswith(('.png', '.jpg', '.jpeg')):
                full_path += '.jpg'  # 默认扩展名

            # 根据路径或 split 划分训练/测试
            if dataset_name == "CUHK-PEDES":
                if 'train_query' in img_path:
                    train_lines.append((full_path, caption, mapped_pid, cam_id))
                elif 'test_query' in img_path:
                    query_lines.append((full_path, caption, mapped_pid, cam_id))
                else:
                    gallery_lines.append((full_path, caption, mapped_pid, cam_id))
            elif dataset_name == "ICFG-PEDES":
                split = item.get('split', 'test')
                if split == 'train':
                    train_lines.append((full_path, caption, mapped_pid, cam_id))
                elif split == 'test':
                    query_lines.append((full_path, caption, mapped_pid, cam_id))
                else:
                    gallery_lines.append((full_path, caption, mapped_pid, cam_id))
            elif dataset_name == "RSTPReid":
                split = item.get('split', 'test')
                if split == 'train':
                    train_lines.append((full_path, caption, mapped_pid, cam_id))
                elif split == 'test':
                    query_lines.append((full_path, caption, mapped_pid, cam_id))
                else:
                    gallery_lines.append((full_path, caption, mapped_pid, cam_id))

        # 根据 is_train 参数返回
        if hasattr(args, 'is_train') and args.is_train:
            list_lines_all.extend(train_lines)
        else:
            list_lines_all.extend(query_lines + gallery_lines)
            args.query_data = query_lines
            args.gallery_data = gallery_lines

    args.global_pid_list = global_pid_list
    return list_lines_all


class DataBuilder_t2i:
    def __init__(self, args, is_distributed=False):
        self.args = args
        self.is_distributed = is_distributed
        self.dataset_configs = args.dataset_configs  # 期望为包含多个数据集配置的列表

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

        # 定义变换
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
