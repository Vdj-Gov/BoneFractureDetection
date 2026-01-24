from ultralytics import YOLO 
import cv2 
import os 
from pathlib import Path
import torch
import sys
import numpy as np

# Check Python environment
venv_path = os.path.join(os.path.dirname(sys.executable), '.venv')
if '.venv' in sys.executable or 'venv' in sys.executable.lower():
    print("⚠️  WARNING: Running from .venv environment")
    print("   For best compatibility, use conda environment: conda activate YoloV8")
    print("   Or update ultralytics in .venv: pip install --upgrade ultralytics\n")

# Fix for PyTorch 2.6+ compatibility with model loading
# PyTorch 2.6+ requires explicit safe globals for loading model weights
print("Setting up PyTorch 2.6+ compatibility...")
try:
    from ultralytics.nn.tasks import DetectionModel
    from torch.nn.modules.container import Sequential, ModuleList, ModuleDict
    from torch.nn import Module
    
    # Add all necessary safe globals for PyTorch 2.6+
    safe_globals = [
        DetectionModel,
        Sequential,
        ModuleList,
        ModuleDict,
        Module,
    ]
    
    # Add common torch modules that might be in the checkpoint
    try:
        from torch.nn import Conv2d, BatchNorm2d, ReLU, SiLU, Upsample, MaxPool2d, AdaptiveAvgPool2d
        safe_globals.extend([Conv2d, BatchNorm2d, ReLU, SiLU, Upsample, MaxPool2d, AdaptiveAvgPool2d])
    except:
        pass
    
    # Add safe globals
    torch.serialization.add_safe_globals(safe_globals)
    print("✓ PyTorch 2.6+ compatibility: Safe globals added successfully\n")
except Exception as e:
    print(f"⚠️  Warning: Could not add safe globals: {e}")
    print("   If model loading fails, try:")
    print("   1. Activate conda environment: conda activate YoloV8")
    print("   2. Update ultralytics: pip install --upgrade ultralytics\n")

# Get the base directory (where this script is located)
base_dir = Path(__file__).resolve().parent

def show_preprocessing_comparison(image_name, split='test', save_comparison=True, display=True):
    """
 
    
    
    """
    # Construct paths
    original_path = base_dir / "dataset" / split / "images" / image_name
    preprocessed_path = base_dir / "preprocessing_output" / split / "preprocessed_images" / image_name
    
    # Check if files exist
    if not original_path.exists():
        print(f"Error: Original image not found at {original_path}")
        return None
    
    if not preprocessed_path.exists():
        print(f"Error: Preprocessed image not found at {preprocessed_path}")
        print(f"Make sure you've run Step01-Preprocess.py first to generate preprocessed images.")
        return None
    
    # Load images
    img_original = cv2.imread(str(original_path))
    img_preprocessed = cv2.imread(str(preprocessed_path), cv2.IMREAD_GRAYSCALE)
    
    if img_original is None:
        print(f"Error: Could not load original image from {original_path}")
        return None
    
    if img_preprocessed is None:
        print(f"Error: Could not load preprocessed image from {preprocessed_path}")
        return None
    
    # Debug: Print image statistics
    print(f"\nPreprocessed image stats (before enhancement):")
    print(f"  Min: {img_preprocessed.min()}, Max: {img_preprocessed.max()}")
    print(f"  Mean: {img_preprocessed.mean():.2f}, Std: {img_preprocessed.std():.2f}")
    
    # Apply contrast enhancement to preprocessed image for better visualization
    # The preprocessed image might be normalized and hard to see
    p_min, p_max = img_preprocessed.min(), img_preprocessed.max()
    p_range = p_max - p_min
    
    if p_range < 100:  # Low contrast image
        # Apply contrast stretching
        if p_range > 0:
            img_preprocessed = ((img_preprocessed.astype(np.float32) - p_min) / p_range * 255).astype(np.uint8)
            print(f"  Applied contrast stretching (range was {p_min}-{p_max})")
        else:
            # All pixels are the same value - apply histogram equalization
            img_preprocessed = cv2.equalizeHist(img_preprocessed)
            print(f"  Applied histogram equalization (all pixels were {p_min})")
    else:
        # Already has good contrast, but apply slight enhancement for visibility
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_preprocessed = clahe.apply(img_preprocessed)
        print(f"  Applied CLAHE enhancement for better visibility")
    
    print(f"  After enhancement - Min: {img_preprocessed.min()}, Max: {img_preprocessed.max()}")
    
    # Convert original to grayscale for fair comparison (preprocessed is grayscale)
    img_original_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    
    # Resize images to same height if they differ
    h1, w1 = img_original_gray.shape
    h2, w2 = img_preprocessed.shape
    
    if h1 != h2 or w1 != w2:
        # Resize preprocessed to match original dimensions
        img_preprocessed = cv2.resize(img_preprocessed, (w1, h1), interpolation=cv2.INTER_LINEAR)
        print(f"Note: Resized preprocessed image to match original dimensions ({w1}x{h1})")
    
    # Ensure both images are uint8 and properly scaled
    img_original_gray = img_original_gray.astype(np.uint8)
    img_preprocessed = img_preprocessed.astype(np.uint8)
    
    # Add labels to images
    img_original_labeled = img_original_gray.copy()
    img_preprocessed_labeled = img_preprocessed.copy()
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (255, 255, 255)  # White text
    
    # Get text size for centering
    (text_width_orig, text_height_orig), _ = cv2.getTextSize("Original", font, font_scale, thickness)
    (text_width_prep, text_height_prep), _ = cv2.getTextSize("Preprocessed", font, font_scale, thickness)
    
    # Add background rectangles for text
    cv2.rectangle(img_original_labeled, (10, 10), (10 + text_width_orig + 10, 10 + text_height_orig + 10), (0, 0, 0), -1)
    cv2.rectangle(img_preprocessed_labeled, (10, 10), (10 + text_width_prep + 10, 10 + text_height_prep + 10), (0, 0, 0), -1)
    
    # Add text
    cv2.putText(img_original_labeled, "Original", (15, 15 + text_height_orig), 
                font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(img_preprocessed_labeled, "Preprocessed", (15, 15 + text_height_prep), 
                font, font_scale, color, thickness, cv2.LINE_AA)
    
    # Combine images side by side
    combined = np.hstack([img_original_labeled, img_preprocessed_labeled])
    
    # Save if requested
    if save_comparison:
        output_dir = base_dir / "output"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"preprocessing_comparison_{split}_{Path(image_name).stem}.png"
        cv2.imwrite(str(output_path), combined)
        print(f"Comparison saved to: {output_path}")
    
    # Display if requested
    if display:
        # Resize for display if too large
        display_height = 800
        h, w = combined.shape
        if h > display_height:
            scale = display_height / h
            new_w = int(w * scale)
            combined_display = cv2.resize(combined, (new_w, display_height), interpolation=cv2.INTER_LINEAR)
        else:
            combined_display = combined
        
        # Print final stats for debugging
        print(f"\nFinal combined image stats:")
        print(f"  Shape: {combined_display.shape}")
        print(f"  Min: {combined_display.min()}, Max: {combined_display.max()}")
        print(f"  Mean: {combined_display.mean():.2f}")
        
        window_name = f"Original vs Preprocessed - {image_name}"
        cv2.imshow(window_name, combined_display)
        print(f"\nDisplaying comparison window: '{window_name}'")
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)
    
    return combined

def show_all_comparisons(image_name, split='test', img_original=None, img_predict=None, img_truth=None, save_comparison=True, display=True):
    """
    Display all 4 images (Original, Preprocessed, Prediction, Truth) in a 2x2 grid
    
    Args:
        image_name: Name of the image file
        split: Dataset split ('train', 'valid', or 'test')
        img_original: Original image (BGR format) - if None, will load from dataset
        img_predict: Prediction image with bounding boxes - if None, will be skipped
        img_truth: Truth image with bounding boxes - if None, will be skipped
        save_comparison: Whether to save the comparison image
        display: Whether to display the comparison using cv2.imshow
    """
    # Load preprocessed image
    preprocessed_path = base_dir / "preprocessing_output" / split / "preprocessed_images" / image_name
    
    if not preprocessed_path.exists():
        print(f"Warning: Preprocessed image not found at {preprocessed_path}")
        print(f"Showing only available images...")
        img_preprocessed = None
    else:
        img_preprocessed = cv2.imread(str(preprocessed_path), cv2.IMREAD_GRAYSCALE)
        if img_preprocessed is not None:
            # Apply contrast enhancement for visibility
            p_min, p_max = img_preprocessed.min(), img_preprocessed.max()
            p_range = p_max - p_min
            if p_range < 100:
                if p_range > 0:
                    img_preprocessed = ((img_preprocessed.astype(np.float32) - p_min) / p_range * 255).astype(np.uint8)
                else:
                    img_preprocessed = cv2.equalizeHist(img_preprocessed)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                img_preprocessed = clahe.apply(img_preprocessed)
            # Convert to BGR for display
            img_preprocessed = cv2.cvtColor(img_preprocessed, cv2.COLOR_GRAY2BGR)
    
    # Prepare images
    images_to_show = []
    labels = []
    
    # Original
    if img_original is not None:
        images_to_show.append(img_original.copy())
        labels.append("Original")
    elif preprocessed_path.exists():
        # Load original if not provided
        original_path = base_dir / "dataset" / split / "images" / image_name
        if original_path.exists():
            img_orig = cv2.imread(str(original_path))
            if img_orig is not None:
                images_to_show.append(img_orig)
                labels.append("Original")
    
    # Preprocessed
    if img_preprocessed is not None:
        images_to_show.append(img_preprocessed)
        labels.append("Preprocessed")
    
    # Prediction
    if img_predict is not None:
        images_to_show.append(img_predict.copy())
        labels.append("Prediction")
    
    # Truth
    if img_truth is not None:
        images_to_show.append(img_truth.copy())
        labels.append("Ground Truth")
    
    if len(images_to_show) == 0:
        print("Error: No images to display")
        return None
    
    # Resize all images to the same size (use the smallest dimensions)
    min_h = min(img.shape[0] for img in images_to_show)
    min_w = min(img.shape[1] for img in images_to_show)
    
    # Resize all images
    resized_images = []
    for img in images_to_show:
        if img.shape[0] != min_h or img.shape[1] != min_w:
            img_resized = cv2.resize(img, (min_w, min_h), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = img.copy()
        
        # Add label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        color = (255, 255, 255)
        label = labels[len(resized_images)]
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(img_resized, (10, 10), (10 + text_width + 10, 10 + text_height + 10), (0, 0, 0), -1)
        cv2.putText(img_resized, label, (15, 15 + text_height), font, font_scale, color, thickness, cv2.LINE_AA)
        resized_images.append(img_resized)
    
    # Create 2x2 grid
    if len(resized_images) == 1:
        combined = resized_images[0]
    elif len(resized_images) == 2:
        combined = np.hstack(resized_images)
    elif len(resized_images) == 3:
        # Put first two on top, third centered below
        top_row = np.hstack([resized_images[0], resized_images[1]])
        # Create a blank image for the third
        blank = np.zeros_like(resized_images[0])
        bottom_row = np.hstack([resized_images[2], blank])
        combined = np.vstack([top_row, bottom_row])
    else:  # 4 images
        top_row = np.hstack([resized_images[0], resized_images[1]])
        bottom_row = np.hstack([resized_images[2], resized_images[3]])
        combined = np.vstack([top_row, bottom_row])
    
    # Save if requested
    if save_comparison:
        output_dir = base_dir / "output"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"all_comparisons_{split}_{Path(image_name).stem}.png"
        cv2.imwrite(str(output_path), combined)
        print(f"All comparisons saved to: {output_path}")
    
    # Display if requested
    if display:
        display_height = 1000
        h, w = combined.shape[:2]
        if h > display_height:
            scale = display_height / h
            new_w = int(w * scale)
            combined_display = cv2.resize(combined, (new_w, display_height), interpolation=cv2.INTER_LINEAR)
        else:
            combined_display = combined
        
        window_name = f"All Comparisons - {image_name}"
        cv2.imshow(window_name, combined_display)
        print(f"\nDisplaying all comparisons in window: '{window_name}'")
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)
    
    return combined

def compare_predictions_original_vs_preprocessed(image_name, split='test', model=None, threshold=0.5, save_comparison=True, display=True):
    """
    Compare model predictions on original vs preprocessed images
    
    Args:
        image_name: Name of the image file
        split: Dataset split ('train', 'valid', or 'test')
        model: YOLO model instance (if None, will load from default path)
        threshold: Confidence threshold for predictions
        save_comparison: Whether to save the comparison image
        display: Whether to display the comparison using cv2.imshow
    
    Returns:
        Dictionary with predictions on both images and comparison stats
    """
    print(f"\n{'='*60}")
    print(f"Starting prediction comparison for: {image_name}")
    print(f"Split: {split}, Threshold: {threshold}")
    print(f"{'='*60}")
    
    # Load model if not provided
    if model is None:
        model_path = base_dir / "runs" / "train" / "My-Model2" / "weights" / "best.pt"
        if not model_path.exists():
            print(f"Error: Model not found at {model_path}")
            return None
        model = YOLO(str(model_path))
    
    # Load original image
    original_path = base_dir / "dataset" / split / "images" / image_name
    if not original_path.exists():
        print(f"Error: Original image not found at {original_path}")
        return None
    
    img_original = cv2.imread(str(original_path))
    if img_original is None:
        print(f"Error: Could not load original image")
        return None
    
    H, W, _ = img_original.shape
    
    # Load preprocessed image
    preprocessed_path = base_dir / "preprocessing_output" / split / "preprocessed_images" / image_name
    if not preprocessed_path.exists():
        print(f"Warning: Preprocessed image not found at {preprocessed_path}")
        print(f"Make sure you've run Step01-Preprocess.py first.")
        return None
    
    img_preprocessed = cv2.imread(str(preprocessed_path), cv2.IMREAD_GRAYSCALE)
    if img_preprocessed is None:
        print(f"Error: Could not load preprocessed image")
        return None
    
    # Convert preprocessed to BGR for model (YOLO expects 3-channel)
    img_preprocessed_bgr = cv2.cvtColor(img_preprocessed, cv2.COLOR_GRAY2BGR)
    
    # Resize preprocessed to match original if needed
    h2, w2 = img_preprocessed_bgr.shape[:2]
    if h2 != H or w2 != W:
        img_preprocessed_bgr = cv2.resize(img_preprocessed_bgr, (W, H), interpolation=cv2.INTER_LINEAR)
    
    print("\n" + "="*60)
    print("Running predictions on ORIGINAL image...")
    print("="*60)
    # Predict on original
    results_original = model(img_original, conf=threshold, verbose=False)[0]
    img_predict_original = img_original.copy()
    boxes_original = []
    
    for result in results_original.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        boxes_original.append((x1, y1, x2, y2, score, int(class_id)))
        
        class_name = results_original.names[int(class_id)].upper()
        cv2.rectangle(img_predict_original, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_predict_original, f"{class_name} {score:.2f}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    
    print(f"Found {len(boxes_original)} detections on original image")
    
    print("\n" + "="*60)
    print("Running predictions on PREPROCESSED image...")
    print("="*60)
    # Predict on preprocessed
    results_preprocessed = model(img_preprocessed_bgr, conf=threshold, verbose=False)[0]
    img_predict_preprocessed = img_preprocessed_bgr.copy()
    boxes_preprocessed = []
    
    for result in results_preprocessed.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        boxes_preprocessed.append((x1, y1, x2, y2, score, int(class_id)))
        
        class_name = results_preprocessed.names[int(class_id)].upper()
        cv2.rectangle(img_predict_preprocessed, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_predict_preprocessed, f"{class_name} {score:.2f}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    
    print(f"Found {len(boxes_preprocessed)} detections on preprocessed image")
    
    # Print comparison statistics
    print("\n" + "="*60)
    print("PREDICTION COMPARISON STATISTICS")
    print("="*60)
    print(f"Original image detections: {len(boxes_original)}")
    if boxes_original:
        avg_conf_orig = sum(b[4] for b in boxes_original) / len(boxes_original)
        print(f"  Average confidence: {avg_conf_orig:.3f}")
        print(f"  Confidence range: {min(b[4] for b in boxes_original):.3f} - {max(b[4] for b in boxes_original):.3f}")
    
    print(f"\nPreprocessed image detections: {len(boxes_preprocessed)}")
    if boxes_preprocessed:
        avg_conf_prep = sum(b[4] for b in boxes_preprocessed) / len(boxes_preprocessed)
        print(f"  Average confidence: {avg_conf_prep:.3f}")
        print(f"  Confidence range: {min(b[4] for b in boxes_preprocessed):.3f} - {max(b[4] for b in boxes_preprocessed):.3f}")
    
    print(f"\nDifference: {len(boxes_preprocessed) - len(boxes_original)} detections")
    if len(boxes_original) > 0 and len(boxes_preprocessed) > 0:
        conf_diff = avg_conf_prep - avg_conf_orig
        print(f"Confidence difference: {conf_diff:+.3f} ({'preprocessed' if conf_diff > 0 else 'original'} has higher confidence)")
    
    # Create side-by-side comparison
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color = (255, 255, 255)
    
    img_orig_labeled = img_predict_original.copy()
    img_prep_labeled = img_predict_preprocessed.copy()
    
    # Original label
    (text_w1, text_h1), _ = cv2.getTextSize(f"Original ({len(boxes_original)} detections)", font, font_scale, thickness)
    cv2.rectangle(img_orig_labeled, (10, 10), (10 + text_w1 + 10, 10 + text_h1 + 10), (0, 0, 0), -1)
    cv2.putText(img_orig_labeled, f"Original ({len(boxes_original)} detections)", (15, 15 + text_h1), 
               font, font_scale, color, thickness, cv2.LINE_AA)
    
    # Preprocessed label
    (text_w2, text_h2), _ = cv2.getTextSize(f"Preprocessed ({len(boxes_preprocessed)} detections)", font, font_scale, thickness)
    cv2.rectangle(img_prep_labeled, (10, 10), (10 + text_w2 + 10, 10 + text_h2 + 10), (0, 0, 0), -1)
    cv2.putText(img_prep_labeled, f"Preprocessed ({len(boxes_preprocessed)} detections)", (15, 15 + text_h2), 
               font, font_scale, color, thickness, cv2.LINE_AA)
    
    # Combine side by side
    combined = np.hstack([img_orig_labeled, img_prep_labeled])
    
    # Save if requested
    output_path = None
    if save_comparison:
        output_dir = base_dir / "output"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"prediction_comparison_original_vs_preprocessed_{Path(image_name).stem}.png"
        cv2.imwrite(str(output_path), combined)
        print(f"\n✓ Comparison saved to: {output_path}")
    
    # Display if requested
    if display:
        try:
            # Close any existing windows first
            cv2.destroyAllWindows()
            
            display_height = 800
            h, w = combined.shape[:2]
            print(f"\nCombined image shape: {h}x{w}")
            
            if h > display_height:
                scale = display_height / h
                new_w = int(w * scale)
                combined_display = cv2.resize(combined, (new_w, display_height), interpolation=cv2.INTER_LINEAR)
                print(f"Resized for display to: {display_height}x{new_w}")
            else:
                combined_display = combined
            
            # Ensure image is uint8
            if combined_display.dtype != np.uint8:
                combined_display = combined_display.astype(np.uint8)
            
            window_name = "Predictions: Original vs Preprocessed"
            print(f"\n{'='*60}")
            print(f"DISPLAYING COMPARISON WINDOW")
            print(f"{'='*60}")
            print(f"Window name: '{window_name}'")
            print(f"Image shape: {combined_display.shape}")
            print(f"Image dtype: {combined_display.dtype}")
            print(f"Image min/max: {combined_display.min()}/{combined_display.max()}")
            
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, combined_display)
            print(f"\n✓ Window displayed successfully!")
            print("Press any key in the window to close it...")
            
            # Wait for key press
            key = cv2.waitKey(0) & 0xFF
            print(f"Key pressed (code: {key}). Closing window...")
            
            cv2.destroyWindow(window_name)
            print("✓ Window closed successfully.")
            
        except Exception as e:
            print(f"\n⚠️  Error displaying window: {e}")
            print("The comparison image has been saved. Please check the output folder.")
            if output_path:
                print(f"Saved image location: {output_path}")
            import traceback
            traceback.print_exc()
    
    # Print final summary
    print(f"\n{'='*60}")
    print("PREDICTION COMPARISON COMPLETE")
    print(f"{'='*60}")
    print(f"✓ Original image: {len(boxes_original)} detections")
    print(f"✓ Preprocessed image: {len(boxes_preprocessed)} detections")
    if output_path:
        print(f"✓ Comparison saved to: {output_path}")
    print(f"{'='*60}\n")
    
    return {
        'original_boxes': boxes_original,
        'preprocessed_boxes': boxes_preprocessed,
        'original_image': img_predict_original,
        'preprocessed_image': img_predict_preprocessed,
        'combined': combined,
        'saved_path': output_path
    }

# Use a test image from the dataset
test_image_name = "z_0_206_png.rf.2d08385a45054aee7124a5e5937e88df.jpg"
test_label_name = "z_0_206_png.rf.2d08385a45054aee7124a5e5937e88df.txt"

imgTest = base_dir / "dataset" / "test" / "images" / test_image_name
imgAnot = base_dir / "dataset" / "test" / "labels" / test_label_name

img = cv2.imread(str(imgTest))
if img is None:
    print(f"Error: Could not load image from {imgTest}")
    exit(1)
    
H , W, _ = img.shape 


# Predict :

imgPredict = img.copy() 

# Use the trained model
model_path = base_dir / "runs" / "train" / "My-Model2" / "weights" / "best.pt"

#load the model 
model = YOLO(str(model_path))

threshold = 0.5 

results = model(imgPredict)[0]

for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score , class_id = result

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    if score > threshold :
        cv2.rectangle(imgPredict, (x1,y1), (x2,y2), (0,255,0), 1)

        class_name = results.names[int(class_id)].upper()

        cv2.putText(imgPredict, class_name, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,255,0), 1, cv2.LINE_AA)


# Ground Truth

imgTruth = img.copy()

with open(str(imgAnot),'r') as file:
    lines = file.readlines()


annotations = [] 

for line in lines :
    values = line.split()
    label = values[0]
    x, y, w, h = map(float, values[1:])
    annotations.append((label, x, y, w, h))

for annotation in annotations:
    label , x, y, w, h, = annotation

    label = results.names[int(label)].upper()

    x1 = int((x - w / 2) * W)
    y1 = int((y - h / 2) * H)
    x2 = int((x + w / 2) * W)
    y2 = int((y + h / 2) * H)

    # Draw bounding box 
    cv2.rectangle(imgTruth, (x1,y1), (x2, y2), (0,255,0), 1)

    # Display label
    cv2.putText(imgTruth, label,(x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,255,0), 1)

# Create output directory if it doesn't exist
output_dir = base_dir / "output"
output_dir.mkdir(exist_ok=True)

cv2.imwrite(str(output_dir / "imgTruth.png"), imgTruth)
cv2.imwrite(str(output_dir / "imgPredict.png"), imgPredict)
cv2.imwrite(str(output_dir / "imgOriginal.png"), img)
print(f"Images saved to: {output_dir}")

# Show all comparisons in one view (Original, Preprocessed, Prediction, Truth)
print("\n" + "="*60)
print("Showing all comparisons (Original, Preprocessed, Prediction, Truth)...")
print("="*60)
show_all_comparisons(test_image_name, split='test', 
                     img_original=img, 
                     img_predict=imgPredict, 
                     img_truth=imgTruth,
                     save_comparison=True, 
                     display=True)

# Also show individual windows (optional - comment out if you only want the combined view)
print("\n" + "="*60)
print("Showing individual windows...")
print("="*60)
cv2.imshow("Image Predict", imgPredict)
cv2.imshow("Image Truth", imgTruth)
cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Show preprocessing comparison separately
print("\n" + "="*60)
print("Showing preprocessing comparison (Original vs Preprocessed)...")
print("="*60)
show_preprocessing_comparison(test_image_name, split='test', save_comparison=True, display=True)

# Compare predictions on original vs preprocessed images
print("\n" + "="*60)
print("COMPARING PREDICTIONS: Original vs Preprocessed Images")
print("="*60)
try:
    comparison_result = compare_predictions_original_vs_preprocessed(
        test_image_name, 
        split='test', 
        model=model,  # Use the already loaded model
        threshold=threshold,
        save_comparison=True, 
        display=True
    )
    
    if comparison_result is None:
        print("\n⚠️  Warning: Comparison function returned None. Check for errors above.")
    else:
        print("\n✓ Comparison completed successfully!")
        if 'saved_path' in comparison_result and comparison_result['saved_path']:
            print(f"  Saved to: {comparison_result['saved_path']}")
except Exception as e:
    print(f"\n❌ Error running comparison: {e}")
    import traceback
    traceback.print_exc()
