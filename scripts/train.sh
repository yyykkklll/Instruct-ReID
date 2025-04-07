#!/bin/sh
ARCH=$1        # 模型架构，例如 "t2i_reid"
NUM_GPUs=$2    # GPU 数量
DESC=$3        # 实验描述，例如 "t2i_reid_cuhk_icfg_rstp"
SEED=0         # 随机种子

if [[ $# -eq 4 ]]; then
  port=${4}
else
  port=23456
fi

ROOT=../../
export PYTHONPATH=$ROOT:$PYTHONPATH

GLOG_vmodule=MemcachedClient=-1

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node ${NUM_GPUs} --master_port ${port} \
  examples/train_joint.py \
  -a ${ARCH} \
  --seed ${SEED} \
  -b 32 \
  -j 4 \
  --warmup-step 1000 \
  --lr 0.00035 \
  --optimizer Adam \
  --weight-decay 0.0005 \
  --scheduler step_lr \
  --milestones 7 14 \
  --epochs 20 \
  --port ${port} \
  --logs-dir logs/${ARCH}-${DESC} \
  --config scripts/config_cuhk_pedes.yaml \
  --dataset-configs \
    "{'name': 'CUHK-PEDES', 'list_file': 'CUHK-PEDES/train_list.txt', 'root': 'CUHK-PEDES/imgs', 'json_file': 'CUHK-PEDES/annotations/caption_all.json', 'query_list': 'CUHK-PEDES/test_query_list.txt', 'gallery_list': 'CUHK-PEDES/test_gallery_list.txt'}" \
    "{'name': 'ICFG-PEDES', 'list_file': 'ICFG-PEDES/train_list.txt', 'root': 'ICFG-PEDES/imgs', 'json_file': 'ICFG-PEDES/ICFG-PEDES.json', 'query_list': 'ICFG-PEDES/test_query_list.txt', 'gallery_list': 'ICFG-PEDES/test_gallery_list.txt'}" \
    "{'name': 'RSTPReid', 'list_file': 'RSTPReid/train_list.txt', 'root': 'RSTPReid/imgs', 'json_file': 'RSTPReid/data_captions.json', 'query_list': 'RSTPReid/test_query_list.txt', 'gallery_list': 'RSTPReid/test_gallery_list.txt'}" \
  --validate \
  --root <your_project_root> \
  --fp16