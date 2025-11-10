#!/bin/bash
# Improved Drowsiness Detection Training Script
# With regularization to prevent overfitting and ensure good quantization

python train.py \
  --model ai85cdnet \
  --dataset cats_vs_dogs \
  --data data/drowsiness_split \
  --epochs 30 \
  --optimizer Adam \
  --lr 0.001 \
  --batch-size 32 \
  --qat-policy policies/qat_policy.yaml \
  --validation-split 0.1 \
  --print-freq 10 \
  --device MAX78000 \
  --weight-decay 0.0001 \
  --save-sample 10 \
  --confusion \
  --embedding \
  --pr-curves \
  -8
