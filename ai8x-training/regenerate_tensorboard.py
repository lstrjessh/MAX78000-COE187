"""
Regenerate TensorBoard event files with correct labels from existing checkpoint
"""
import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pydoc import locate

# Import training modules
import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import ai8x

# Configuration
CHECKPOINT_PATH = "logs/2025.11.10-235510/best.pth.tar"
DATA_DIR = "data/drowsiness_split"
OUTPUT_LOG_DIR = "logs/tensorboard_regenerated"
BATCH_SIZE = 32

class SimpleArgs:
    """Simple args class for normalization"""
    def __init__(self):
        self.act_mode_8bit = True
        self.truncate_testset = False

def load_model_and_checkpoint(checkpoint_path):
    """Load the model and checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Initialize device (required by ai8x) - MAX78000 = device 85
    ai8x.set_device(device=85, simulate=False, round_avg=False)
    
    # Dynamically load model module (same way train.py does it)
    model_module = locate('models.ai85net-cd')
    model_fn = model_module.ai85cdnet
    
    # Create model using the factory function
    model = model_fn(num_classes=2, dimensions=(128, 128), num_channels=3, bias=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    return model, checkpoint

def get_test_loader(data_dir):
    """Create test data loader"""
    args = SimpleArgs()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        ai8x.normalize(args=args),
    ])
    
    # Import the dataset module
    from datasets.cats_vs_dogs import CatsvsDogs
    
    test_dataset = CatsvsDogs(root_dir=data_dir, d_type='test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    return test_loader, test_dataset

def evaluate_and_log(model, test_loader, writer, class_names):
    """Evaluate model and log results to TensorBoard"""
    all_preds = []
    all_labels = []
    total_loss = 0.0
    correct = 0
    total = 0
    
    criterion = torch.nn.CrossEntropyLoss()
    
    print("Evaluating model on test set...")
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch [{batch_idx + 1}/{len(test_loader)}]")
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(test_loader)
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"  Average Loss: {avg_loss:.4f}")
    
    # Log scalar metrics
    writer.add_scalar('Test/Accuracy', accuracy, 0)
    writer.add_scalar('Test/Loss', avg_loss, 0)
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Create confusion matrix figure
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar=True, square=True, annot_kws={"size": 14})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix - Test Set', fontsize=14)
    plt.tight_layout()
    
    # Convert matplotlib figure to numpy array for TensorBoard
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # Log confusion matrix image (HWC format, convert to CHW for tensorboard)
    writer.add_image('Test/ConfusionMatrix', img_array, 0, dataformats='HWC')
    
    plt.close(fig)
    
    return accuracy, avg_loss

def main():
    # Load model
    model, checkpoint = load_model_and_checkpoint(CHECKPOINT_PATH)
    
    # Get test data
    test_loader, test_dataset = get_test_loader(DATA_DIR)
    
    # Class names from dataset
    class_names = test_dataset.labels
    print(f"Class names: {class_names}")
    
    # Create TensorBoard writer
    os.makedirs(OUTPUT_LOG_DIR, exist_ok=True)
    writer = SummaryWriter(OUTPUT_LOG_DIR)
    
    # Evaluate and log
    accuracy, avg_loss = evaluate_and_log(model, test_loader, writer, class_names)
    
    # Add checkpoint info as text
    writer.add_text('Info/Checkpoint', CHECKPOINT_PATH, 0)
    writer.add_text('Info/Epoch', str(checkpoint.get('epoch', 'Unknown')), 0)
    writer.add_text('Info/BestTop1', f"{checkpoint.get('extras', {}).get('best_top1', 'Unknown'):.2f}%", 0)
    
    writer.close()
    
    print(f"\n✅ TensorBoard files generated successfully!")
    print(f"   Output directory: {OUTPUT_LOG_DIR}")
    print(f"\nTo view in TensorBoard, run:")
    print(f"   tensorboard --logdir=logs")
    print(f"   Then open: http://localhost:6006/")

if __name__ == '__main__':
    main()
