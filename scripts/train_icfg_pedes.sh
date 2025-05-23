#!/bin/bash

# 设置使用的 GPU（仅使用第7张显卡，编号为06）
export CUDA_VISIBLE_DEVICES=6

# ICFG-PEDES
DATASET_CONFIGS='{"name": "ICFG-PEDES", "root": "data/ICFG-PEDES/imgs", "json_file": "data/ICFG-PEDES/annotations/caption_all.json", "cloth_json": "data/ICFG-PEDES/annotations/caption_cloth.json", "id_json": "data/ICFG-PEDES/annotations/caption_id.json"}'

# 运行ICFG-PEDES训练命令
python scripts/train.py \
  --config configs/config_cuhk_pedes.yaml \
  --root data/ICFG-PEDES \
  --dataset-configs "${DATASET_CONFIGS}" \
  --batch-size 128 \
  --workers 0 \
  --fp16 \
  --logs-dir logs/icfg_pedes
