#!/bin/sh
#Genereted Log folder during the Training
LOG_DIRECTORY="../ai8x-training/logs/2023.03.24-201120_cats_vs_dogs"

# Quantization for cats and dogs
python quantize.py $LOG_DIRECTORY/qat_best.pth.tar $LOG_DIRECTORY/qat_best-quantized.pth.tar --device MAX78000 -v "$@"

# Quantization for Key Word Spotted
#python quantize.py ../ai8x-training/logs/2023.04.06-172201_kws/qat_best.pth.tar ../ai8x-training/logs/2023.04.06-172201_kws/qat_best-q.pth.tar --device MAX78000 -v "$@"

# Quantization for Emotion Rcognition 

# Common Template for Quantize Scripts
# python quantize.py ..ai8x-training/logs/<log folder name of your trained model>/qat_best.pth.tar ../ai8x-training/logs/<log folder name of your trained model>/qat_best-q.pth.tar --device MAX78000 -v "$@"
# 2023.04.06-172201_kws

