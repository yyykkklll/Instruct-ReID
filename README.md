# 项目介绍

本模型使用了 **VIT + Bert** 实现了基于文本指导的Reid

文件组织结构（**tree.txt**）：

```python
v3/
├── data/                       # 数据集相关文件
│   ├── CUHK-PEDES/            # 数据集1
│   │   ├── imgs/              # 图像文件
│   │   │   ├── cam_a/
│   │   │   ├── CUHK03/
│   │   │   ├── Market/
│   │   │   ├── test_query/
│   │   │   └── train_query/
│   │   ├── annotations/       # 标注文件
│   │   │   └── caption_all.json
│   │   └── readme.txt         # 数据集说明
│   ├── ICFG-PEDES/            # 数据集2
│   │   ├── imgs/
│   │   │   ├── test/
│   │   │   └── train/
│   │   ├── annotations/
│   │   │   └── ICFG-PEDES.json
│   │   └── processed_data/    # 预处理数据
│   │       ├── data_message
│   │       ├── ind2word.pkl
│   │       ├── test_save.pkl
│   │       └── train_save.pkl
│   └── RSTPReid/              # 数据集3
│       ├── imgs/
│       └── annotations/
│           └── data_captions.json
├── src/                       # 源代码
│   ├── datasets/              # 数据加载与预处理
│   │   ├── data_builder_t2i.py
│   │   ├── image_layer.py
│   │   ├── base_dataset.py
│   │   ├── preprocessor_t2i.py
│   │   ├── sampler.py
│   │   ├── transforms.py
│   │   └── __init__.py
│   ├── models/                # 模型定义
│   │   ├── backbone/          # 模型骨干网络
│   │   │   ├── pass_vit.py
│   │   │   ├── vit_albef.py
│   │   │   ├── vit_albef_ori.py
│   │   │   ├── ckpt.py
│   │   │   └── __init__.py
│   │   ├── pass_transformer_joint.py
│   │   ├── tokenization_bert.py
│   │   ├── xbert.py
│   │   └── __init__.py
│   ├── loss/                  # 损失函数
│   │   ├── adv_loss.py
│   │   └── __init__.py
│   ├── trainer/               # 训练逻辑
│   │   ├── base_trainer.py
│   │   ├── pass_trainer.py
│   │   ├── pass_trainer_joint.py
│   │   └── __init__.py
│   ├── evaluation/            # 评估逻辑
│   │   ├── evaluators_t.py
│   │   └── __init__.py
│   ├── utils/                 # 工具函数
│   │   ├── comm_.py
│   │   ├── distributed_utils.py
│   │   ├── distributed_utils_pt.py
│   │   ├── logging.py
│   │   ├── lr_scheduler.py
│   │   ├── meters.py
│   │   ├── osutils.py
│   │   ├── serialization.py
│   │   ├── vit_rollout.py
│   │   └── __init__.py
│   └── multi_tasks_utils/     # 多任务相关工具（可保留以备扩展）
│       ├── multi_task_distributed_utils.py
│       ├── multi_task_distributed_utils_pt.py
│       └── __init__.py
├── scripts/                   # 可执行脚本
│   ├── train.py               # 训练脚本（重命名自train_joint.py）
│   ├── evaluate.py            # 评估脚本
│   ├── test.py                # 测试脚本（重命名自test_joint.py）
│   ├── check_dataset.py       # 数据检查脚本
│   └── check_checkpoint.py    # 检查点检查脚本
├── configs/                   # 配置文件
│   └── config_cuhk_pedes.yaml
├── pretrained/                # 预训练模型
│   ├── config.json
│   ├── vit-base-patch16-224.pth
│   ├── bert-base-uncased/     # BERT预训练权重
│   │   ├── added_tokens.json
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   ├── special_tokens_map.json
│   │   └── vocab.txt
│   └── vit-base-patch16-224/  # ViT预训练权重
│       ├── config.json
│       ├── preprocessor_config.json
│       ├── pytorch_model.bin
│       └── tf_model.h5
├── logs/                      # 日志和输出
└── README.md                  # 项目说明文档
```

其中**pretrained**文件中的权重文件和预训练模型可前往[google/vit-base-patch16-224 · Hugging Face](https://huggingface.co/google/vit-base-patch16-224)进行下载，或者通过**Baidupan**分享的文件：链接: https://pan.baidu.com/s/1kxKxPmp3QWEf6IugfqnMBg 提取码: 1ukx下载

本模型数据集使用了**CUHK-PEDES,ICFG-PEDES,RSTPReid**进行测试评估.自行前往官网进行下载：

**CUHK-PEDES：**[layumi/Image-Text-Embedding: TOMM2020 Dual-Path Convolutional Image-Text Embedding with Instance Loss https://arxiv.org/abs/1711.05535](https://github.com/layumi/Image-Text-Embedding)

**ICFG-PEDES：**[zifyloo/SSAN: Code of SSAN](https://github.com/zifyloo/SSAN)

**RSTPReid：**[NjtechCVLab/RSTPReid-Dataset: RSTPReid Dataset for Text-based Person Retrieval.](https://github.com/NjtechCVLab/RSTPReid-Dataset)

准备工作完成之后，可使用以下命令进行训练：

```python
CUHK-PEDES:
CUDA_VISIBLE_DEVICES=6 python scripts/train.py --config configs/config_cuhk_pedes.yaml --root data/CUHK-PEDES --dataset-configs "{'name': 'CUHK-PEDES', 'root': 'data/CUHK-PEDES/imgs', 'json_file': 'data/CUHK-PEDES/annotations/caption_all.json'}" --batch-size 128 --workers 0 --fp16 --logs-dir logs/cuhk_pedes


ICFG-PEDES:
CUDA_VISIBLE_DEVICES=6 python scripts/train.py --config configs/config_cuhk_pedes.yaml --root data/ICFG-PEDES --dataset-configs "{'name': 'ICFG-PEDES', 'root': 'data/ICFG-PEDES/imgs', 'json_file': 'data/ICFG-PEDES/annotations/ICFG-PEDES.json'}" --batch-size 128 --workers 0 --fp16 --logs-dir logs/icfg_pedes

RSTPReid:
CUDA_VISIBLE_DEVICES=6 python scripts/train.py --config configs/config_cuhk_pedes.yaml --root data/RSTPReid --dataset-configs "{'name': 'RSTPReid', 'root': 'data/RSTPReid/imgs', 'json_file': 'data/RSTPReid/annotations/data_captions.json'}" --batch-size 128 --workers 0 --fp16 --logs-dir logs/rstp_reid
```

模型训练完成会自动运行评估脚本。也可通过以下命令自行评估测试：

**Linux**

```python
CUHK-PEDES:
CUDA_VISIBLE_DEVICES=6 python scripts/evaluate.py --config configs/config_cuhk_pedes.yaml --root data/CUHK-PEDES --dataset-configs "{'name': 'CUHK-PEDES', 'root': 'data/CUHK-PEDES/imgs', 'json_file': 'data/CUHK-PEDES/annotations/caption_all.json'}" --checkpoint logs/cuhk_pedes/checkpoint_epoch_final.pth --batch-size 128 --workers 0 --fp16

ICFG-PEDES:
CUDA_VISIBLE_DEVICES=6 python scripts/evaluate.py --config configs/config_cuhk_pedes.yaml --root data/ICFG-PEDES --dataset-configs "{'name': 'ICFG-PEDES', 'root': 'data/ICFG-PEDES/imgs', 'json_file': 'data/ICFG-PEDES/ICFG-PEDES.json'}" --checkpoint logs/icfg_pedes/checkpoint_epoch_final.pth --batch-size 128 --workers 0 --fp16

RSTPReid:
CUDA_VISIBLE_DEVICES=6 python scripts/evaluate.py --config configs/config_cuhk_pedes.yaml --root data/RSTPReid --dataset-configs "{'name': 'RSTPReid', 'root': 'data/RSTPReid/imgs', 'json_file': 'data/RSTPReid/data_captions.json'}" --checkpoint logs/rstp_reid/checkpoint_epoch_final.pth --batch-size 128 --workers 0 --fp16
```

**Windows**

```python
CUHK-PEDES:
set CUDA_VISIBLE_DEVICES=6 && python scripts/evaluate.py --config configs\config_cuhk_pedes.yaml --root data\CUHK-PEDES --dataset-configs "{'name': 'CUHK-PEDES', 'root': 'data\CUHK-PEDES\imgs', 'json_file': 'data\CUHK-PEDES\annotations\caption_all.json'}" --checkpoint logs\cuhk_pedes\checkpoint_epoch_final.pth --batch-size 128 --workers 0 --fp16

ICFG-PEDES:
set CUDA_VISIBLE_DEVICES=6 && python scripts/evaluate.py --config configs\config_cuhk_pedes.yaml --root data\ICFG-PEDES --dataset-configs "{'name': 'ICFG-PEDES', 'root': 'data\ICFG-PEDES\imgs', 'json_file': 'data\ICFG-PEDES\ICFG-PEDES.json'}" --checkpoint logs\icfg_pedes\checkpoint_epoch_final.pth --batch-size 128 --workers 0 --fp16

RSTPReid:
set CUDA_VISIBLE_DEVICES=6 && python scripts/evaluate.py --config configs\config_cuhk_pedes.yaml --root data\RSTPReid --dataset-configs "{'name': 'RSTPReid', 'root': 'data\RSTPReid\imgs', 'json_file': 'data\RSTPReid\data_captions.json'}" --checkpoint logs\rstp_reid\checkpoint_epoch_final.pth --batch-size 128 --workers 0 --fp16
```

相关参数配置可在**configs/config_cuhk_pedes.yaml**文件中调整