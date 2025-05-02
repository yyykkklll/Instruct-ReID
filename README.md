# 基于文本指导的 ReID 模型

本项目基于 **VIT + Bert + 门控融合机制 +  身份衣物接纠缠**架构，实现了 **基于文本指导的行人重识别（ReID）**。通过结合视觉和文本信息，模型在多个公开数据集上展现了优异的性能。

---

## 文件组织结构

```plaintext
v3/
├── data/                       # 数据集相关文件
│   ├── CUHK-PEDES/            # 数据集1
│   │   ├── imgs/              # 图像文件
│   │   │   ├── cam\_a/
│   │   │   ├── CUHK03/
│   │   │   ├── Market/
│   │   │   ├── test\_query/
│   │   │   └── train\_query/
│   │   ├── annotations/       # 标注文件
│   │   │   └── caption\_all.json
│   │   └── readme.txt         # 数据集说明
│   ├── ICFG-PEDES/            # 数据集2
│   │   ├── imgs/
│   │   │   ├── test/
│   │   │   └── train/
│   │   ├── annotations/
│   │   │   └── ICFG-PEDES.json
│   │   └── processed\_data/    # 预处理数据
│   │       ├── data\_message
│   │       ├── ind2word.pkl
│   │       ├── test\_save.pkl
│   │       └── train\_save.pkl
│   └── RSTPReid/              # 数据集3
│       ├── imgs/
│       └── annotations/
│           └── data\_captions.json
├── src/                       # 源代码
│   ├── datasets/              # 数据加载与预处理
│   │   ├── data\_builder\_t2i.py
│   │   ├── transforms.py
│   │   └── init.py
│   ├── models/                # 模型定义
│   │   ├── pass\_transformer\_joint.py
│   │   └── **init**.py
│   ├── loss/                  # 损失函数
│   │   ├── adv\_loss.py
│   │   └── **init**.py
│   ├── trainer/               # 训练逻辑
│   │   ├── pass\_trainer\_joint.py
│   │   └── **init**.py
│   ├── evaluation/            # 评估逻辑
│   │   ├── evaluators\_t.py
│   │   └── **init**.py
│   ├── utils/                 # 工具函数
│   │   ├── comm\_.py
│   │   ├── distributed\_utils.py
│   │   ├── distributed\_utils\_pt.py
│   │   ├── logging.py
│   │   ├── lr\_scheduler.py
│   │   ├── meters.py
│   │   ├── osutils.py
│   │   ├── serialization.py
│   │   ├── vit\_rollout.py
│   │   └── **init**.py
├── scripts/                   # 可执行脚本
│   ├── train.py               # 训练脚本（重命名自train\_joint.py）
│   ├── evaluate.py            # 评估脚本
│   ├── test.py                # 测试脚本（重命名自test\_joint.py）
│   ├── check\_dataset.py       # 数据检查脚本
│   └── check\_checkpoint.py    # 检查点检查脚本
├── configs/                   # 配置文件
│   └── config\_cuhk\_pedes.yaml
├── pretrained/                # 预训练模型
│   ├── config.json
│   ├── vit-base-patch16-224.pth
│   ├── bert-base-uncased/     # BERT预训练权重
│   │   ├── added\_tokens.json
│   │   ├── config.json
│   │   ├── pytorch\_model.bin
│   │   ├── special\_tokens\_map.json
│   │   └── vocab.txt
│   └── vit-base-patch16-224/  # ViT预训练权重
│       ├── config.json
│       ├── preprocessor\_config.json
│       ├── pytorch\_model.bin
│       └── tf\_model.h5
├── logs/                      # 日志和输出
└── README.md                  # 项目说明文档
```

> **注意**：预训练模型权重可通过以下方式获取：
> - [Hugging Face: google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)
> - 或通过百度网盘下载：链接: [https://pan.baidu.com/s/1kxKxPmp3QWEf6IugfqnMBg](https://pan.baidu.com/s/1kxKxPmp3QWEf6IugfqnMBg) 提取码: `1ukx`

---

## 准备工作

在正式训练模型之前，请确保完成以下准备工作：

### 环境依赖

请参考 `requirements.txt` 文件安装所有依赖项：

```plaintext
accelerate==1.0.1
clip==0.2.0
easydict==1.13
matplotlib==3.7.5
opencv-python==4.9.0.80
pillow==10.4.0
protobuf==5.29.4
psutil==7.0.0
PyYAML==6.0.1
requests==2.32.3
scikit-learn==1.3.0
scipy==1.10.1
sympy==1.13.3
tensorboardX==2.6.2.2
timm==0.9.16
tqdm==4.65.0
transformers==4.46.3
h5py==3.11.0
safetensors==0.5.3
tokenizers==0.20.3
ftfy==5.8
huggingface-hub==0.29.3
Jinja2==3.1.6
networkx==3.1
numpy==1.24.4
```

#### 安装步骤

1. **安装 PyTorch**（根据需要选择 CUDA 版本）：

   ```bash
   pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
   ```

2. **安装其他依赖**：

   ```bash
   pip install -r requirements.txt
   ```

### 数据集准备

本项目支持以下三个数据集进行训练和评估：

- **CUHK-PEDES**: [layumi/Image-Text-Embedding](https://github.com/layumi/Image-Text-Embedding)
- **ICFG-PEDES**: [zifyloo/SSAN](https://github.com/zifyloo/SSAN)
- **RSTPReid**: [NjtechCVLab/RSTPReid-Dataset](https://github.com/NjtechCVLab/RSTPReid-Dataset)

请自行前往上述链接下载并解压数据集到 `data/` 目录下。

---

## 使用方法

### 训练模型

使用以下命令启动训练过程：

```bash
# CUHK-PEDES
CUDA_VISIBLE_DEVICES=6 python scripts/train.py --config configs/config_cuhk_pedes.yaml --root data/CUHK-PEDES --dataset-configs "{\"name\": \"CUHK-PEDES\", \"root\": \"data/CUHK-PEDES/imgs\", \"json_file\": \"data/CUHK-PEDES/annotations/caption_all.json\", \"cloth_json\": \"data/CUHK-PEDES/annotations/caption_cloth.json\", \"id_json\": \"data/CUHK-PEDES/annotations/caption_id.json\"}" --batch-size 128 --workers 0 --fp16 --logs-dir logs/cuhk_pedes

# ICFG-PEDES
CUDA_VISIBLE_DEVICES=6 python scripts/train.py --config configs/config_cuhk_pedes.yaml --root data/ICFG-PEDES --dataset-configs "{\"name\": \"ICFG-PEDES\", \"root\": \"data/ICFG-PEDES\", \"json_file\": \"data/ICFG-PEDES/annotations/ICFG-PEDES.json\", \"cloth_json\": \"data/ICFG-PEDES/annotations/caption_cloth.json\", \"id_json\": \"data/ICFG-PEDES/annotations/caption_id.json\"}" --batch-size 128 --workers 0 --fp16 --logs-dir logs/icfg_pedes

# RSTPReid
CUDA_VISIBLE_DEVICES=6 python scripts/train.py --config configs/config_cuhk_pedes.yaml --root data/RSTPReid --dataset-configs "{\"name\": \"RSTPReid\", \"root\": \"data/RSTPReid\", \"json_file\": \"data/RSTPReid/annotations/data_captions.json\", \"cloth_json\": \"data/RSTPReid/annotations/caption_cloth.json\", \"id_json\":\"data/RSTPReid/annotations/caption_id.json\"}" --batch-size 128 --workers 0 --fp16 --logs-dir logs/rstp_reid
```

训练完成后，模型会自动运行评估脚本。

---

### 模型评估

如果需要手动评估模型，可以使用以下命令：

```bash
# CUHK-PEDES
CUDA_VISIBLE_DEVICES=6 python scripts/evaluate.py --config configs/config_cuhk_pedes.yaml --root data/CUHK-PEDES --dataset-configs "{\"name\": \"CUHK-PEDES\", \"root\": \"data/CUHK-PEDES/imgs\", \"json_file\": \"data/CUHK-PEDES/annotations/caption_all.json\", \"cloth_json\": \"data/CUHK-PEDES/annotations/caption_cloth.json\", \"id_json\": \"data/CUHK-PEDES/annotations/caption_id.json\"}" --checkpoint logs/cuhk_pedes/checkpoint_epoch_final.pth --batch-size 128 --workers 0 --fp16 --logs-dir logs/cuhk_pedes

# ICFG-PEDES 和 RSTPReid 类似，仅需调整路径和参数。
```

---

### 配置文件

所有超参数和配置均可在 `configs/config_cuhk_pedes.yaml` 文件中调整。请根据需求修改对应参数。

---

## 注意事项

1. **GPU 设备**：请确保正确设置 `CUDA_VISIBLE_DEVICES` 参数以指定可用 GPU。
2. **混合精度训练**：`--fp16` 参数启用混合精度训练，可显著加速训练过程。
3. **日志目录**：训练和评估结果默认存储在 `logs/` 目录下。

---

