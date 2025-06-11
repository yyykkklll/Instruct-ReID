import json
from pathlib import Path
import random
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import logging
import os


def merge_sub_datasets(dataset_configs, args, skip_logging=False):
    """
    合并多个数据集,划分训练、查询和图库,确保训练集和测试集无交集
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
        
        # 验证文件存在性
        missing_files = []
        for file in [json_file, cloth_json_file, id_json_file]:
            if not file.exists():
                missing_files.append(str(file))
        
        if missing_files:
            logging.error(f"Missing files for {dataset_name}: {', '.join(missing_files)}")
            continue

        # 加载JSON文件
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                attr_dict_raw = json.load(f)
            
            with open(cloth_json_file, 'r', encoding='utf-8') as f:
                cloth_dict = {str(item['id']): item['captions'] for item in json.load(f)}
            
            with open(id_json_file, 'r', encoding='utf-8') as f:
                id_dict = {str(item['id']): item['captions'] for item in json.load(f)}
                
            if not skip_logging:
                logging.info(f"Loaded {len(attr_dict_raw)} items, {len(cloth_dict)} cloth captions, "
                             f"and {len(id_dict)} ID captions for {dataset_name}")
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"Failed to parse JSON files: {e}")
            continue

        # 处理数据项
        items_by_id = {}
        invalid_pids = []
        valid_images = 0
        
        for item in attr_dict_raw:
            img_path = item.get('file_path', item.get('img_path', ''))
            pid = str(item.get('id', item.get('pid', 0)))
            
            # 验证PID格式
            try:
                pid_int = int(pid)
                if not (0 <= pid_int < 1000000):
                    invalid_pids.append(pid)
                    continue
            except ValueError:
                invalid_pids.append(pid)
                continue

            # 获取描述文本
            cloth_captions = cloth_dict.get(pid, [])
            id_captions = id_dict.get(pid, [])
            cloth_caption = cloth_captions[0] if cloth_captions else ""
            id_caption = id_captions[0] if id_captions else ""

            # 构建完整路径
            full_path = prefix / img_path
            if not full_path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp'):
                full_path = full_path.with_suffix('.jpg')
            
            if not full_path.exists():
                continue

            valid_images += 1
            cam_id = "0"
            is_matched = 1
            
            if pid not in items_by_id:
                items_by_id[pid] = []
            items_by_id[pid].append((str(full_path), cloth_caption, id_caption, pid, cam_id, is_matched))

        # 记录处理结果
        if invalid_pids:
            logging.warning(f"Found {len(invalid_pids)} invalid PIDs in {dataset_name}")
        if not skip_logging:
            logging.info(f"Processed {valid_images} valid images for {dataset_name}")
        if not items_by_id:
            logging.warning(f"No valid items found for {dataset_name}")
            continue

        # 划分数据集
        all_ids = list(items_by_id.keys())
        random.seed(0)
        random.shuffle(all_ids)
        
        train_ratio = 0.85
        train_id_count = int(len(all_ids) * train_ratio)
        train_ids = all_ids[:train_id_count]
        test_ids = all_ids[train_id_count:]
        
        # 准备负样本的cloth_caption映射
        if hasattr(args, 'is_train') and args.is_train:
            pid_to_cloth = {
                pid: items_by_id[pid][0][1] for pid in items_by_id
                if items_by_id[pid][0][1]  # 确保cloth_caption非空
            }
            other_cloths = list(pid_to_cloth.values())
            
            if not other_cloths:
                logging.error(f"No valid cloth captions found for negative sampling in {dataset_name}")
                continue

        # 处理训练数据
        if hasattr(args, 'is_train') and args.is_train:
            train_image_count = 0
            
            for pid in train_ids:
                # 映射全局PID
                if pid not in global_pid_list:
                    global_pid_list[pid] = global_pid_counter
                    global_pid_counter += 1
                mapped_pid = global_pid_list[pid]
                
                for item in items_by_id[pid]:
                    # 添加正样本
                    list_lines_all.append(item[:3] + (mapped_pid,) + item[4:])
                    train_image_count += 1
                    
                    # 添加负样本
                    neg_cloth = random.choice(other_cloths)
                    list_lines_all.append(
                        (item[0], neg_cloth, item[2], mapped_pid, item[4], 0)
                        )
                    train_image_count += 1
            
            logging.info(f"{dataset_name} - Added to training set: {train_image_count} samples")
        # 处理测试数据
        else:
            # 确保50% ID重叠（原代码中1%不合理，改为30%）
            overlap_ratio = 0.3
            overlap_id_count = int(len(test_ids) * overlap_ratio)
            overlap_ids = test_ids[:overlap_id_count]
            non_overlap_ids = test_ids[overlap_id_count:]
            
            # 非重叠ID平均分配到查询和图库
            split_point = len(non_overlap_ids) // 2
            query_only_ids = non_overlap_ids[:split_point]
            gallery_only_ids = non_overlap_ids[split_point:]
            
            query_lines = []
            gallery_lines = []
            
            # 处理重叠ID
            for pid in overlap_ids:
                if pid not in global_pid_list:
                    global_pid_list[pid] = global_pid_counter
                    global_pid_counter += 1
                mapped_pid = global_pid_list[pid]
                
                items = items_by_id[pid]
                random.shuffle(items)
                split_idx = max(1, len(items) // 2)
                
                # 添加到查询集
                for item in items[:split_idx]:
                    query_lines.append(item[:3] + (mapped_pid,) + item[4:])
                
                # 添加到图库集
                for item in items[split_idx:]:
                    gallery_lines.append(item[:3] + (mapped_pid,) + item[4:])
            
            # 处理查询集独有ID
            for pid in query_only_ids:
                if pid not in global_pid_list:
                    global_pid_list[pid] = global_pid_counter
                    global_pid_counter += 1
                mapped_pid = global_pid_list[pid]
                
                for item in items_by_id[pid]:
                    query_lines.append(item[:3] + (mapped_pid,) + item[4:])
            
            # 处理图库集独有ID
            for pid in gallery_only_ids:
                if pid not in global_pid_list:
                    global_pid_list[pid] = global_pid_counter
                    global_pid_counter += 1
                mapped_pid = global_pid_list[pid]
                
                for item in items_by_id[pid]:
                    gallery_lines.append(item[:3] + (mapped_pid,) + item[4:])
            
            # 验证训练集和测试集无图像交集
            train_images = {item[0] for item in list_lines_all}
            test_images = {item[0] for item in query_lines + gallery_lines}
            image_overlap = train_images & test_images
            
            if image_overlap:
                overlapping_files = "\n".join(list(image_overlap)[:5])
                logging.error(f"Found {len(image_overlap)} overlapping images. Examples:\n{overlapping_files}")
                raise ValueError("Train and test sets must have no image overlap")
            
            # 保存测试数据
            list_lines_all.extend(query_lines + gallery_lines)
            
            if not hasattr(args, 'query_data'):
                args.query_data = []
            if not hasattr(args, 'gallery_data'):
                args.gallery_data = []
            
            args.query_data.extend(query_lines)
            args.gallery_data.extend(gallery_lines)
            
            # 记录ID重叠情况
            # query_ids = {item[3] for item in query_lines}
            # gallery_ids = {item[3] for item in gallery_lines}
            # overlap_ids = query_ids & gallery_ids
            # overlap_ratio = len(overlap_ids) / len(query_ids) * 100 if query_ids else 0

    # 更新全局状态
    args.num_classes = global_pid_counter
    args.global_pid_list = global_pid_list
    
    # 验证PID范围
    if list_lines_all:
        pids = {item[3] for item in list_lines_all}
        min_pid, max_pid = min(pids), max(pids)
        
        if min_pid < 0 or max_pid >= args.num_classes:
            raise ValueError(
                f"Invalid PID range: min={min_pid}, max={max_pid}, "
                f"expected [0, {args.num_classes - 1}]")

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
        self.image_cache = {}
        self.failed_images = set()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, cloth_caption, id_caption, pid, cam_id, is_matched = self.data[index]
        
        # 验证PID范围
        if pid < 0 or pid >= self.args.num_classes:
            logging.error(f"Invalid PID {pid} at index {index}, img_path: {img_path}")
            pid = 0  # 使用默认PID避免崩溃

        # 尝试从缓存加载
        if img_path in self.image_cache:
            image = self.image_cache[img_path]
        elif img_path in self.failed_images:
            image = self._generate_dummy_image()
        else:
            try:
                # 使用更健壮的图像加载方式
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"Image not found: {img_path}")
                
                image = Image.open(img_path).convert('RGB')
                
                # 验证图像有效性
                if image.width <= 0 or image.height <= 0:
                    raise ValueError(f"Invalid image dimensions: {image.width}x{image.height}")
                
                self.image_cache[img_path] = image
            except Exception as e:
                logging.warning(f"Failed to load image {img_path}: {e}")
                self.failed_images.add(img_path)
                image = self._generate_dummy_image()

        if self.transform is not None:
            image = self.transform(image)

        # 确保文本描述有效
        cloth_caption = cloth_caption if cloth_caption else "no description"
        id_caption = id_caption if id_caption else "no id description"

        return (image, cloth_caption, id_caption, 
                torch.tensor(pid, dtype=torch.long),
                torch.tensor(int(cam_id), dtype=torch.long), 
                torch.tensor(is_matched, dtype=torch.long))
    
    def _generate_dummy_image(self):
        """生成替代图像"""
        return Image.fromarray(
            (torch.rand(3, self.args.height, self.args.width).numpy().transpose(1, 2, 0) * 255).astype('uint8')
        ).convert('RGB')
    


class DataBuilder_t2i:
    """
    T2I-ReID 数据构建器,负责加载和划分数据集
    """
    def __init__(self, args, is_distributed=False):
        self.args = args
        self.is_distributed = is_distributed
        self.dataset_configs = args.dataset_configs
        
        # 图像预处理
        self.transform_train = transforms.Compose([
            transforms.Resize((args.height, args.width)),
            transforms.RandomCrop((args.height, args.width), padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.transform_test = transforms.Compose([
            transforms.Resize((args.height, args.width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def get_num_classes(self):
        """获取全局PID数量"""
        self.args.is_train = True
        # 跳过日志获取类别数
        merge_sub_datasets(self.dataset_configs, self.args, skip_logging=True)
        return self.args.num_classes

    def _load_data(self, list_lines):
        """转换数据格式并验证"""
        return [
            (img_path, cloth_caption, id_caption, int(pid), int(cam_id), int(is_matched))
            for img_path, cloth_caption, id_caption, pid, cam_id, is_matched in list_lines
        ]

    def build_dataset(self, data, is_train=False):
        """构建数据集实例"""
        transform = self.transform_train if is_train else self.transform_test
        return T2IReIDDataset(data, self.args, transform=transform)

    def _build_data_loader(self, dataset, is_train=False):
        """构建数据加载器"""
        def collate_fn(batch):
            images, cloth_captions, id_captions, pids, cam_ids, is_matched = zip(*batch)
            images = torch.stack(images, dim=0)
            pids = torch.stack(pids, dim=0)
            cam_ids = torch.stack(cam_ids, dim=0)
            is_matched = torch.stack(is_matched, dim=0)
            return images, cloth_captions, id_captions, pids, cam_ids, is_matched

        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=is_train,
            num_workers=self.args.workers,
            pin_memory=True,
            drop_last=is_train,
            collate_fn=collate_fn
        )

    def build_data(self, is_train=False):
        """构建数据加载器"""
        self.args.is_train = is_train
        list_lines = merge_sub_datasets(self.dataset_configs, self.args)
        data = self._load_data(list_lines)
        
        if is_train:
            # 训练集统计
            unique_pids = len(set(item[3] for item in data))
            unique_cams = len(set(item[4] for item in data))
            
            logging.info("Dataset statistics:")
            logging.info("  ------------------------------------------")
            logging.info("  subset   | # ids | # images | # cameras")
            logging.info("  ------------------------------------------")
            logging.info(f"  train    | {unique_pids:5d} | {len(data):8d} | {unique_cams:8d}")
            logging.info("  ------------------------------------------")
            
            dataset = self.build_dataset(data, is_train=True)
            return self._build_data_loader(dataset, is_train=True), dataset
        else:
            # 测试集统计
            query_data = self._load_data(self.args.query_data)
            gallery_data = self._load_data(self.args.gallery_data)
            
            query_pids = len(set(item[3] for item in query_data))
            gallery_pids = len(set(item[3] for item in gallery_data))
            query_cams = len(set(item[4] for item in query_data))
            gallery_cams = len(set(item[4] for item in gallery_data))
            
            logging.info("Dataset statistics:")
            logging.info("  ------------------------------------------")
            logging.info("  subset   | # ids | # images | # cameras")
            logging.info("  ------------------------------------------")
            logging.info(f"  query    | {query_pids:5d} | {len(query_data):8d} | {query_cams:8d}")
            logging.info(f"  gallery  | {gallery_pids:5d} | {len(gallery_data):8d} | {gallery_cams:8d}")
            logging.info("  ------------------------------------------")
            
            query_dataset = self.build_dataset(query_data, is_train=False)
            gallery_dataset = self.build_dataset(gallery_data, is_train=False)
            
            return (
                self._build_data_loader(query_dataset, is_train=False),
                self._build_data_loader(gallery_dataset, is_train=False)
            )