# PowerShell version - Quantization script for drowsiness detection model

# Get the latest training log directory path
$LOG_DIRECTORY = "..\ai8x-training\logs\latest_log_dir"

# Resolve the symbolic link if it exists
if (Test-Path $LOG_DIRECTORY) {
    $LOG_DIRECTORY = (Get-Item $LOG_DIRECTORY).Target
    if (-not $LOG_DIRECTORY) {
        # If it's not a symlink, get the actual path
        $LOG_DIRECTORY = (Get-Item "..\ai8x-training\logs\latest_log_dir").FullName
    }
}

Write-Host "Using log directory: $LOG_DIRECTORY"

# Quantize the trained model (using QAT checkpoint)
python quantize.py `
  "$LOG_DIRECTORY/qat_best.pth.tar" `
  "$LOG_DIRECTORY/qat_best_quantized.pth.tar" `
  --device MAX78000 `
  -v `
  @args

Write-Host ""
Write-Host "Quantization complete!" -ForegroundColor Green
Write-Host "Quantized model saved to: $LOG_DIRECTORY/qat_best_quantized.pth.tar"
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Copy the quantized model to ai8x-synthesis/trained/"
Write-Host "2. Run the synthesis tool (ai8xize) to generate C code"
