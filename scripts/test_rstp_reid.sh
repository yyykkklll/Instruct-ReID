#!/bin/bash

# 设置使用的 GPU（仅使用第7张显卡，编号为06）
export CUDA_VISIBLE_DEVICES=6

# RSTPReid
DATASET_CONFIGS='{"name": "RSTPReid", "root": "data/RSTPReid/imgs", "json_file": "data/RSTPReid/annotations/caption_all.json", "cloth_json": "data/RSTPReid/annotations/caption_cloth.json", "id_json": "data/RSTPReid/annotations/caption_id.json"}'

# 运行评估命令
python scripts/evaluate.py \
  --config configs/config_cuhk_pedes.yaml \
  --root data/RSTPReid\
  --dataset-configs "${DATASET_CONFIGS}" \
  --checkpoint logs/rstp_reid/checkpoint_epoch_final.pth \
  --batch-size 128 \
  --workers 0 \
  --fp16 \
  --logs-dir logs/rstp_reid