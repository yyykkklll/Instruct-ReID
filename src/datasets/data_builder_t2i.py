import json
from pathlib import Path
import random
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import logging


def check_overlap(query_data, gallery_data):
    """
    检查查询和图库数据之间的ID重叠比例
    """
    query_set = {item[3] for item in query_data}
    gallery_set = {item[3] for item in gallery_data}
    overlap = query_set & gallery_set
    query_overlap_ratio = len(overlap) / len(query_set) * 100 if query_set else 0
    gallery_overlap_ratio = len(overlap) / len(gallery_set) * 100 if gallery_set else 0
    avg_overlap_ratio = (query_overlap_ratio + gallery_overlap_ratio) / 2
    return avg_overlap_ratio


def merge_sub_datasets(dataset_configs, args, skip_logging=False):
    """
    合并多个数据集,划分训练、查询和图库,确保训练集和测试集无交集,查询和画廊集有50% ID重叠
    """
    list_lines_all = []
    global_pid_list = getattr(args, 'global_pid_list', {})
    global_pid_counter = len(global_pid_list)
    ROOT_DIR = Path(__file__).parent.parent.parent

    if not dataset_configs:
        logging.error("No dataset configurations provided")
        return list_lines_all

    for config in dataset_configs:
        dataset_name = config.get('name', '')
        prefix = ROOT_DIR / config.get('root', '')
        json_file = ROOT_DIR / config.get('json_file', '')
        cloth_json_file = ROOT_DIR / config.get('cloth_json', '')
        id_json_file = ROOT_DIR / config.get('id_json', '')

        if not dataset_name:
            logging.error("Dataset name not specified in config")
            continue

        if not skip_logging:
            logging.info(f"Loading dataset {dataset_name} from {json_file}")
        if not json_file.exists():
            logging.error(f"JSON file not found for {dataset_name} at: {json_file}")
            continue
        if not cloth_json_file.exists():
            logging.error(f"Cloth JSON file not found for {dataset_name} at: {cloth_json_file}")
            continue
        if not id_json_file.exists():
            logging.error(f"ID JSON file not found for {dataset_name} at: {id_json_file}")
            continue

        # 加载 JSON 文件
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                attr_dict_raw = json.load(f)
            if not skip_logging:
                logging.info(f"Loaded {len(attr_dict_raw)} items from {json_file}")
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON file {json_file}: {e}")
            continue

        try:
            with open(cloth_json_file, 'r', encoding='utf-8') as f:
                cloth_dict = {str(item['id']): item['captions'] for item in json.load(f)}
            with open(id_json_file, 'r', encoding='utf-8') as f:
                id_dict = {str(item['id']): item['captions'] for item in json.load(f)}
            if not skip_logging:
                logging.info(f"Loaded {len(cloth_dict)} cloth captions and {len(id_dict)} ID captions for {dataset_name}")
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse cloth/ID JSON file: {e}")
            continue

        # 按ID分组并验证图像
        items_by_id = {}
        valid_images = 0
        invalid_pids = []
        for item in attr_dict_raw:
            img_path = item.get('file_path', item.get('img_path', ''))
            pid = str(item.get('id', item.get('pid', 0)))
            cloth_captions = cloth_dict.get(pid, [])
            id_captions = id_dict.get(pid, [])
            cloth_caption = cloth_captions[0] if cloth_captions else ''
            id_caption = id_captions[0] if id_captions else ''
            ext = '.jpg'

            try:
                pid_int = int(pid)
                if pid_int < 0 or pid_int >= 1000000:
                    invalid_pids.append(pid)
                    continue
            except ValueError:
                invalid_pids.append(pid)
                continue

            cam_id = "0"
            full_path = prefix / img_path
            if not full_path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp'):
                full_path = full_path.with_suffix(ext)

            if not full_path.exists():
                continue

            valid_images += 1
            is_matched = 1  # 默认值,稍后在训练集处理中调整
            if pid not in items_by_id:
                items_by_id[pid] = []
            items_by_id[pid].append((str(full_path), cloth_caption, id_caption, pid, cam_id, is_matched))

        if invalid_pids:
            logging.warning(f"Found {len(invalid_pids)} invalid PIDs in {dataset_name}")
        if not skip_logging:
            logging.info(f"Found {valid_images} valid images for {dataset_name}")
        if not items_by_id:
            logging.warning(f"No valid items found for {dataset_name}")
            continue

        # PID统计
        all_pids = [int(pid) for pid in items_by_id.keys()]


        # 划分数据集
        all_ids = list(items_by_id.keys())
        random.seed(0)
        random.shuffle(all_ids)

        train_ratio = 0.85
        train_id_count = int(len(all_ids) * train_ratio)
        train_ids = all_ids[:train_id_count]
        test_ids = all_ids[train_id_count:]

        if hasattr(args, 'is_train') and args.is_train:
            # 训练集：统一PID映射,并生成正负样本对
            train_image_count = 0
            all_cloth_captions = [(pid, items_by_id[pid][0][1]) for pid in items_by_id.keys()]  # (pid, cloth_caption)
            for pid in train_ids:
                if pid not in global_pid_list:
                    global_pid_list[pid] = global_pid_counter
                    global_pid_counter += 1
                mapped_pid = global_pid_list[pid]
                for item in items_by_id[pid]:
                    # 正样本：使用对应的 cloth_caption
                    pos_cloth_caption = item[1]
                    list_lines_all.append((item[0], pos_cloth_caption, item[2], mapped_pid, item[4], 1))
                    train_image_count += 1

                    # 负样本：随机选择其他身份的 cloth_caption
                    other_pids = [p for p in all_cloth_captions if p[0] != pid]
                    if other_pids:
                        neg_cloth_caption = random.choice(other_pids)[1]
                        list_lines_all.append((item[0], neg_cloth_caption, item[2], mapped_pid, item[4], 0))
                        train_image_count += 1

            logging.info(f"{dataset_name} - Added to training set: {train_image_count} samples (with pos/neg pairs)")
        else:
            # 测试集：允许未见PID,继续分配索引
            query_lines = []
            gallery_lines = []

            # 实现30% ID重叠
            overlap_ratio = 0.3
            overlap_id_count = int(len(test_ids) * overlap_ratio)
            overlap_ids = test_ids[:overlap_id_count]
            non_overlap_ids = test_ids[overlap_id_count:]
            query_only_ids = non_overlap_ids[:len(non_overlap_ids) // 2]
            gallery_only_ids = non_overlap_ids[len(non_overlap_ids) // 2:]

            used_pids = set()
            for pid in overlap_ids:
                if pid not in global_pid_list:
                    global_pid_list[pid] = global_pid_counter
                    global_pid_counter += 1
                mapped_pid = global_pid_list[pid]
                items = items_by_id[pid]
                random.shuffle(items)
                split_point = max(1, len(items) // 2)
                query_items = items[:split_point]
                gallery_items = items[split_point:]
                for item in query_items:
                    query_lines.append((item[0], item[1], item[2], mapped_pid, item[4], item[5]))
                    used_pids.add(mapped_pid)
                for item in gallery_items:
                    gallery_lines.append((item[0], item[1], item[2], mapped_pid, item[4], item[5]))
                    used_pids.add(mapped_pid)
            for pid in query_only_ids:
                if pid not in global_pid_list:
                    global_pid_list[pid] = global_pid_counter
                    global_pid_counter += 1
                mapped_pid = global_pid_list[pid]
                for item in items_by_id[pid]:
                    query_lines.append((item[0], item[1], item[2], mapped_pid, item[4], item[5]))
                    used_pids.add(mapped_pid)
            for pid in gallery_only_ids:
                if pid not in global_pid_list:
                    global_pid_list[pid] = global_pid_counter
                    global_pid_counter += 1
                mapped_pid = global_pid_list[pid]
                for item in items_by_id[pid]:
                    gallery_lines.append((item[0], item[1], item[2], mapped_pid, item[4], item[5]))
                    used_pids.add(mapped_pid)

            # 验证训练集和测试集无图像交集
            train_images = {item[0] for item in list_lines_all}
            test_images = {item[0] for item in query_lines + gallery_lines}
            image_overlap = train_images & test_images
            if image_overlap:
                logging.error(f"Found {len(image_overlap)} overlapping images between train and test sets")
                raise ValueError("Train and test sets must have no image overlap")

            list_lines_all.extend(query_lines + gallery_lines)
            if not hasattr(args, 'query_data'):
                args.query_data = []
            if not hasattr(args, 'gallery_data'):
                args.gallery_data = []
            args.query_data.extend(query_lines)
            args.gallery_data.extend(gallery_lines)

            common_ids = set(x[3] for x in query_lines) & set(x[3] for x in gallery_lines)
            check_overlap(query_lines, gallery_lines)

    # 设置 num_classes
    args.num_classes = global_pid_counter
    args.global_pid_list = global_pid_list
    if list_lines_all:
        mapped_pids = [x[3] for x in list_lines_all]
        if min(mapped_pids) < 0 or max(mapped_pids) >= args.num_classes:
            raise ValueError(
                f"Invalid PID range: min={min(mapped_pids)}, max={max(mapped_pids)}, expected [0, {args.num_classes - 1}]")

    if not skip_logging:
        logging.info(f"Total data items loaded: {len(list_lines_all)}")
    return list_lines_all


class T2IReIDDataset(Dataset):
    """
    T2I-ReID 数据集类,加载图像、衣物描述、身份描述和匹配标签
    """
    def __init__(self, data, args, transform=None):
        self.data = data
        self.args = args
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, cloth_caption, id_caption, pid, cam_id, is_matched = self.data[index]
        img_path = Path(img_path)

        if pid < 0 or pid >= self.args.num_classes:
            logging.error(f"Invalid PID {pid} at index {index}, img_path: {img_path}")
            raise ValueError(f"PID {pid} out of range [0, {self.args.num_classes - 1}]")

        try:
            # 使用更健壮的图像加载方式
            image = Image.open(img_path).convert('RGB')
            
            # 检查图像是否有效
            if image.width <= 0 or image.height <= 0:
                raise ValueError(f"Invalid image dimensions: {image.width}x{image.height}")
                
        except Exception as e:
            logging.warning(f"Failed to load image {img_path}: {e}")
            # 创建一个随机噪声图像作为替代
            image_array = torch.rand(3, self.args.height, self.args.width)
            image = transforms.ToPILImage()(image_array)

        if self.transform is not None:
            image = self.transform(image)

        # 确保文本描述不为None
        cloth_caption = cloth_caption if cloth_caption else ""
        id_caption = id_caption if id_caption else ""

        return (image, cloth_caption, id_caption, torch.tensor(pid, dtype=torch.long),
                torch.tensor(cam_id, dtype=torch.long), torch.tensor(is_matched, dtype=torch.long))


class DataBuilder_t2i:
    """
    T2I-ReID 数据构建器,负责加载和划分数据集
    """
    def __init__(self, args, is_distributed=False):
        self.args = args
        self.is_distributed = is_distributed
        self.dataset_configs = args.dataset_configs
        
        # ViT预训练模型使用的归一化参数，将像素值从[0,1]映射到[-1,1]
        self.transform_train = transforms.Compose([
            transforms.Resize((args.height, args.width)),
            transforms.RandomCrop((args.height, args.width), padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到[-1,1]范围
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize((args.height, args.width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到[-1,1]范围
        ])

    def get_num_classes(self):
        self.args.is_train = True
        # 在获取类别数时跳过日志打印
        list_lines = merge_sub_datasets(self.dataset_configs, self.args, skip_logging=True)
        return self.args.num_classes

    def _load_data(self, list_lines):
        return [(img_path, cloth_caption, id_caption, int(pid), int(cam_id), int(is_matched))
                for img_path, cloth_caption, id_caption, pid, cam_id, is_matched in list_lines]

    def build_dataset(self, data, is_train=False):
        transform = self.transform_train if is_train else self.transform_test
        return T2IReIDDataset(data, self.args, transform=transform)

    def _build_train_loader(self, data):
        if not data:
            logging.error("Training dataset is empty")
            raise ValueError("Training dataset is empty")
        dataset = self.build_dataset(data, is_train=True)

        def collate_fn(batch):
            images, cloth_captions, id_captions, pids, cam_ids, is_matched = zip(*batch)
            images = torch.stack(images, dim=0)
            pids = torch.stack(pids, dim=0)
            cam_ids = torch.stack(cam_ids, dim=0)
            is_matched = torch.stack(is_matched, dim=0)
            return images, cloth_captions, id_captions, pids, cam_ids, is_matched

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
        if not data:
            logging.error("Test dataset is empty")
            raise ValueError("Test dataset is empty")
        dataset = self.build_dataset(data, is_train=False)

        def collate_fn(batch):
            images, cloth_captions, id_captions, pids, cam_ids, is_matched = zip(*batch)
            images = torch.stack(images, dim=0)
            pids = torch.stack(pids, dim=0)
            cam_ids = torch.stack(cam_ids, dim=0)
            is_matched = torch.stack(is_matched, dim=0)
            return images, cloth_captions, id_captions, pids, cam_ids, is_matched

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
        list_lines = merge_sub_datasets(self.dataset_configs, self.args)
        data = self._load_data(list_lines)
        if is_train:
            logging.info("Dataset statistics:")
            logging.info("  ------------------------------------------")
            logging.info("  subset   | # ids | # images | # cameras")
            logging.info("  ------------------------------------------")
            logging.info(f"  train    | {len(set(item[3] for item in data)):5d} | {len(data):8d} | "
                         f"{len(set(item[4] for item in data)):8d}")
            logging.info("  ------------------------------------------")
            return self._build_train_loader(data)
        else:
            query_data = self._load_data(self.args.query_data)
            gallery_data = self._load_data(self.args.gallery_data)
            logging.info("Dataset statistics:")
            logging.info("  ------------------------------------------")
            logging.info("  subset   | # ids | # images | # cameras")
            logging.info("  ------------------------------------------")
            logging.info(f"  query    | {len(set(item[3] for item in query_data)):5d} | {len(query_data):8d} | "
                         f"{len(set(item[4] for item in query_data)):8d}")
            logging.info(f"  gallery  | {len(set(item[3] for item in gallery_data)):5d} | {len(gallery_data):8d} | "
                         f"{len(set(item[4] for item in gallery_data)):8d}")
            logging.info("  ------------------------------------------")
            return self._build_test_loader(query_data, is_query=True), self._build_test_loader(gallery_data,
                                                                                               is_query=False)