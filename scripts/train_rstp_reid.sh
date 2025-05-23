#!/bin/bash

# 设置使用的 GPU（仅使用第7张显卡，编号为06）
export CUDA_VISIBLE_DEVICES=6

# RSTPReid
DATASET_CONFIGS='{"name": "RSTPReid", "root": "data/RSTPReid/imgs", "json_file": "data/RSTPReid/annotations/data_captions.json", "cloth_json": "data/RSTPReid/annotations/caption_cloth.json", "id_json": "data/RSTPReid/annotations/caption_id.json"}'

# 运行RSTPReid训练命令
python scripts/train.py \
  --config configs/config_cuhk_pedes.yaml \
  --root data/RSTPReid \
  --dataset-configs "${DATASET_CONFIGS}" \
  --batch-size 128 \
  --workers 0 \
  --fp16 \
  --logs-dir logs/rstp_reid