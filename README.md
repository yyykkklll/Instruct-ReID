# 🔍 基于文本指导的 ReID 模型

本项目基于 **ViT + BERT + 门控融合机制 + 身份-衣物解纠缠** 架构，实现了 **基于文本指导的行人重识别（Text-to-Image ReID）**。  
通过结合视觉与语言信息，模型在多个公开数据集上表现优异，具备较强的泛化能力和解释性。

------

## 🗂️ 文件组织结构

```markdown
v3/
├── data/                         # 数据集
│   ├── CUHK-PEDES/               # 数据集 1
│   ├── ICFG-PEDES/               # 数据集 2
│   └── RSTPReid/                 # 数据集 3
├── src/                          # 源代码
│   ├── datasets/                 # 数据加载与预处理
│   ├── models/                   # 模型结构定义
│   ├── loss/                     # 损失函数
│   ├── trainer/                  # 训练流程
│   ├── evaluation/               # 模型评估
│   └── utils/                    # 工具函数
├── scripts/                      # 执行脚本
├── configs/                      # 配置文件
├── pretrained/                   # 预训练权重
├── logs/                         # 训练 & 测试输出
└── README.md                     # 当前文档
```

📎 *详细文件说明见正文下方对应模块*

------

## 🔧 准备工作

### 📦 环境依赖

请先安装 PyTorch（根据 CUDA 版本）：

```bash
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

然后安装其他依赖：

```bash
pip install -r requirements.txt
```

📌 依赖列表见 `requirements.txt`，包括但不限于：

- `transformers`
- `timm`
- `opencv-python`
- `h5py`
- `tensorboardX`
- `scikit-learn`

### 📁 数据集准备

项目支持以下数据集：

| 数据集名称 | 链接                                                         |
| ---------- | ------------------------------------------------------------ |
| CUHK-PEDES | 🔗 [layumi/Image-Text-Embedding](https://github.com/layumi/Image-Text-Embedding) |
| ICFG-PEDES | 🔗 [zifyloo/SSAN](https://github.com/zifyloo/SSAN)            |
| RSTPReid   | 🔗 [NjtechCVLab/RSTPReid-Dataset](https://github.com/NjtechCVLab/RSTPReid-Dataset) |

> 请将下载后的数据集解压到 `data/` 目录下

------

## 🧠 训练与评估

### 🚀 模型训练

以 CUHK-PEDES 为例：

```bash
CUDA_VISIBLE_DEVICES=6 python scripts/train.py \
  --config configs/config_cuhk_pedes.yaml \
  --root data/CUHK-PEDES \
  --dataset-configs "{\"name\": \"CUHK-PEDES\", \"root\": \"data/CUHK-PEDES/imgs\", \"json_file\": \"data/CUHK-PEDES/annotations/caption_all.json\", \"cloth_json\": \"data/CUHK-PEDES/annotations/caption_cloth.json\", \"id_json\": \"data/CUHK-PEDES/annotations/caption_id.json\"}" \
  --batch-size 128 --workers 0 --fp16 --logs-dir logs/cuhk_pedes
```

🔁 ICFG-PEDES 与 RSTPReid 仅需更换参数路径。

### 📊 模型评估

```bash
CUDA_VISIBLE_DEVICES=6 python scripts/evaluate.py \
  --config configs/config_cuhk_pedes.yaml \
  --root data/CUHK-PEDES \
  --dataset-configs "{\"name\": \"CUHK-PEDES\", \"root\": \"data/CUHK-PEDES/imgs\", \"json_file\": \"data/CUHK-PEDES/annotations/caption_all.json\", \"cloth_json\": \"data/CUHK-PEDES/annotations/caption_cloth.json\", \"id_json\": \"data/CUHK-PEDES/annotations/caption_id.json\"}" \
  --checkpoint logs/cuhk_pedes/checkpoint_epoch_final.pth \
  --batch-size 128 --workers 0 --fp16
```

------

## ⚙️ 配置文件说明

配置文件示例路径：

```bash
configs/config_cuhk_pedes.yaml
```

📌 可调整参数：

- 模型结构（例如 BERT 与 ViT 配置）
- 优化器 & 学习率策略
- 批大小与训练周期
- 日志输出路径等

------

## 🧩 预训练模型

本项目依赖以下预训练模型：

| 模型              | 来源             | 权重链接                                                     |
| ----------------- | ---------------- | ------------------------------------------------------------ |
| ViT-B/16          | HuggingFace      | [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) |
| BERT base uncased | HuggingFace      | [bert-base-uncased](https://huggingface.co/bert-base-uncased) |
| 备用百度网盘      | 🔐 提取码: `1ukx` | [pan.baidu.com](https://pan.baidu.com/s/1kxKxPmp3QWEf6IugfqnMBg) |

------

## 📌 注意事项

- ✅ **GPU 设置**：使用 `CUDA_VISIBLE_DEVICES` 指定训练 GPU
- ⚡ **混合精度**：建议添加 `--fp16` 参数加速训练
- 📁 **输出目录**：默认日志输出到 `logs/`，按数据集命名

------

## 📝 鸣谢与引用

本项目参考或基于以下优秀开源项目：

- [Image-Text-Embedding](https://github.com/layumi/Image-Text-Embedding)
- [SSAN](https://github.com/zifyloo/SSAN)
- [RSTPReid-Dataset](https://github.com/NjtechCVLab/RSTPReid-Dataset)

------

## 🤝 联系我们

如有问题欢迎提出 [Issues](https://github.com/your_repo/issues) 或邮件(qlu.ykelong@gmail.com)联系。

------

> 🧭 **你可以一时看不见努力的回报，但请相信，时间会给你答案。**
