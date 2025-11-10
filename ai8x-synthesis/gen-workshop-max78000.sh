#!/bin/sh
DEVICE="MAX78000"
GEN_PATH="C:\MaximSDK\Examples\MAX78000\CNN"
COMMON_ARGS="--device $DEVICE --timer 0 --display-checkpoint --verbose"

#input files
QUANTIZED_MODEL="../ai8x-training/logs/logs_kws-mlp-v3/best-quantized.pth.tar"
YAML="networks/kws-mlp-v1.yaml"
SAMPLE="../ai8x-training/logs/logs_kws-mlp-v3/sample_kws_20.npy"

#gen for kws mlp
python ai8xize.py --test-dir $GEN_PATH --prefix kws-mlp-v4 --overwrite --checkpoint-file $QUANTIZED_MODEL --config-file $YAML --sample-input $SAMPLE --softmax $COMMON_ARGS "$@"
