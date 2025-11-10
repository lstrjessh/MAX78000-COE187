#!/bin/sh
# Quantization script for drowsiness detection model

# Get the latest training log directory
LOG_DIRECTORY="../ai8x-training/logs/latest_log_dir"

# Quantize the trained model (using QAT checkpoint)
python quantize.py \
  $LOG_DIRECTORY/qat_best.pth.tar \
  $LOG_DIRECTORY/qat_best_quantized.pth.tar \
  --device MAX78000 \
  -v \
  "$@"

echo ""
echo "Quantization complete!"
echo "Quantized model saved to: $LOG_DIRECTORY/qat_best_quantized.pth.tar"
echo ""
echo "Next steps:"
echo "1. Copy the quantized model to ai8x-synthesis/trained/"
echo "2. Run the synthesis tool (ai8xize) to generate C code"
