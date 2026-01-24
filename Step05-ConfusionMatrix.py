"""
Generate Confusion Matrix for Bone Fracture Detection Model
Evaluates the model on validation/test set and creates confusion matrix visualization
"""

from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import torch
import sys
import os
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# Check Python environment
venv_path = os.path.join(os.path.dirname(sys.executable), '.venv')
if '.venv' in sys.executable or 'venv' in sys.executable.lower():
    print("⚠️  WARNING: Running from .venv environment")
    print("   For best compatibility, use conda environment: conda activate YoloV8\n")

# Fix for PyTorch 2.6+ compatibility with model loading
print("Setting up PyTorch 2.6+ compatibility...")
try:
    from ultralytics.nn.tasks import DetectionModel
    from torch.nn.modules.container import Sequential, ModuleList, ModuleDict
    from torch.nn import Module
    
    safe_globals = [
        DetectionModel,
        Sequential,
        ModuleList,
        ModuleDict,
        Module,
    ]
    
    try:
        from torch.nn import Conv2d, BatchNorm2d, ReLU, SiLU, Upsample, MaxPool2d, AdaptiveAvgPool2d
        safe_globals.extend([Conv2d, BatchNorm2d, ReLU, SiLU, Upsample, MaxPool2d, AdaptiveAvgPool2d])
    except:
        pass
    
    torch.serialization.add_safe_globals(safe_globals)
    print("✓ PyTorch 2.6+ compatibility: Safe globals added successfully\n")
except Exception as e:
    print(f"⚠️  Warning: Could not add safe globals: {e}\n")

def plot_confusion_matrix_from_yolo(model, data_yaml, split='val', save_path=None):
    """
    Generate confusion matrix using YOLO's built-in validation
    
    Args:
        model: YOLO model instance
        data_yaml: Path to data.yaml file
        split: Dataset split to evaluate ('val' or 'test')
        save_path: Path to save the confusion matrix plot
    """
    print(f"Evaluating model on {split} set...")
    
    # Run validation
    results = model.val(data=str(data_yaml), split=split, save=True, plots=True)
    
    # The confusion matrix is automatically saved by YOLO
    # But we can also create a custom one
    print(f"\n✓ Validation completed!")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    print(f"  Precision: {results.box.mp:.4f}")
    print(f"  Recall: {results.box.mr:.4f}")
    
    return results

def create_custom_confusion_matrix(model, images_dir, labels_dir, class_names, conf_threshold=0.5, iou_threshold=0.45):
    """
    Create a custom confusion matrix by evaluating each image
    
    Args:
        model: YOLO model instance
        images_dir: Directory containing test images
        labels_dir: Directory containing ground truth labels
        class_names: List of class names
        conf_threshold: Confidence threshold for predictions
        iou_threshold: IoU threshold for NMS
    """
    from pathlib import Path
    import cv2
    
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    
    # Get all images
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    
    print(f"\nEvaluating {len(image_files)} images...")
    
    # Storage for predictions and ground truth
    all_predictions = []
    all_ground_truth = []
    
    # Process each image
    for img_path in image_files:
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Get predictions
        results = model(str(img_path), conf=conf_threshold, iou=iou_threshold, verbose=False)
        
        # Get ground truth
        label_path = labels_dir / (img_path.stem + ".txt")
        gt_boxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        gt_boxes.append(class_id)
        
        # Get predicted boxes
        pred_boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                pred_boxes.append(class_id)
        
        # For binary classification (fracture/no fracture)
        # We'll use: 0 = No Fracture, 1 = Fracture
        has_gt_fracture = len(gt_boxes) > 0
        has_pred_fracture = len(pred_boxes) > 0
        
        all_ground_truth.append(1 if has_gt_fracture else 0)
        all_predictions.append(1 if has_pred_fracture else 0)
    
    # Create confusion matrix
    cm = confusion_matrix(all_ground_truth, all_predictions)
    
    return cm, all_ground_truth, all_predictions

def plot_confusion_matrix(cm, class_names, save_path=None, normalize=False):
    """
    Plot confusion matrix with nice visualization
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        save_path: Path to save the plot
        normalize: Whether to normalize the confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Confusion matrix saved to: {save_path}")
    
    plt.show()
    
    # Print classification report
    print("\n" + "="*60)
    print("Classification Report")
    print("="*60)
    
    # Calculate metrics manually
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
        
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nMetrics:")
        print(f"  True Positives (TP):  {TP}")
        print(f"  True Negatives (TN):  {TN}")
        print(f"  False Positives (FP): {FP}")
        print(f"  False Negatives (FN): {FN}")
        print(f"\n  Accuracy:  {accuracy:.8f}")
        print(f"  Precision: {precision:.8f}")
        print(f"  Recall:    {recall:.8f}")
        print(f"  F1-Score:  {f1_score:.f}")
    else:
        print(f"\nConfusion Matrix:\n{cm}")
    
    print("="*60)

def main():
    """Main function"""
    base_dir = Path(__file__).resolve().parent
    
    print("="*60)
    print("Bone Fracture Detection - Confusion Matrix Generator")
    print("="*60)
    
    # Load model
    model_path = base_dir / "runs" / "train" / "My-Model2" / "weights" / "best.pt"
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using Step03-Train.py")
        return
    
    print(f"\nLoading model from: {model_path}")
    model = YOLO(str(model_path))
    print("✓ Model loaded successfully")
    
    # Get class names
    class_names = model.names
    print(f"Classes: {class_names}")
    
    # Method 1: Use YOLO's built-in validation (recommended)
    data_yaml = base_dir / "data.yaml"
    if data_yaml.exists():
        print(f"\n{'='*60}")
        print("Method 1: Using YOLO's built-in validation")
        print(f"{'='*60}")
        results = plot_confusion_matrix_from_yolo(model, data_yaml, split='val')
        
        # The confusion matrix is saved in the runs directory
        cm_path = base_dir / "runs" / "detect" / "val" / "confusion_matrix.png"
        if cm_path.exists():
            print(f"\n✓ YOLO confusion matrix saved to: {cm_path}")
    
    # Method 2: Create custom confusion matrix for test set
    test_images_dir = base_dir / "dataset" / "test" / "images"
    test_labels_dir = base_dir / "dataset" / "test" / "labels"
    
    if test_images_dir.exists() and test_labels_dir.exists():
        print(f"\n{'='*60}")
        print("Method 2: Creating custom confusion matrix for test set")
        print(f"{'='*60}")
        
        cm, y_true, y_pred = create_custom_confusion_matrix(
            model, 
            test_images_dir, 
            test_labels_dir,
            class_names,
            conf_threshold=0.5,
            iou_threshold=0.45
        )
        
        # Create output directory
        output_dir = base_dir / "evaluation_output"
        output_dir.mkdir(exist_ok=True)
        
        # Plot confusion matrix (not normalized)
        plot_confusion_matrix(
            cm, 
            ['No Fracture', 'Fracture'],
            save_path=output_dir / "confusion_matrix_test.png",
            normalize=False
        )
        
        # Plot normalized confusion matrix
        plot_confusion_matrix(
            cm, 
            ['No Fracture', 'Fracture'],
            save_path=output_dir / "confusion_matrix_test_normalized.png",
            normalize=True
        )
    
    print(f"\n{'='*60}")
    print("Confusion matrix generation completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

