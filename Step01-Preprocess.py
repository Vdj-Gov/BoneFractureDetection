"""
Data Preprocessing Script for Bone Fracture Detection
- Normalizes image data
- Creates histograms for data analysis
- Applies preprocessing techniques (contrast enhancement, noise reduction, etc.)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import shutil
from tqdm import tqdm
import pandas as pd

def normalize_image(img, method='min_max'):
    """
    Normalize image pixel values
    
    Args:
        img: Input image (numpy array)
        method: 'min_max' (0-1), 'z_score' (mean=0, std=1), or 'unit_vector'
    
    Returns:
        Normalized image
    """
    if method == 'min_max':
        # Normalize to [0, 1]
        img_normalized = img.astype(np.float32) / 255.0
    elif method == 'z_score':
        # Z-score normalization (mean=0, std=1)
        mean = np.mean(img)
        std = np.std(img)
        if std > 0:
            img_normalized = (img.astype(np.float32) - mean) / std
        else:
            img_normalized = img.astype(np.float32)
    elif method == 'unit_vector':
        # L2 normalization
        norm = np.linalg.norm(img)
        if norm > 0:
            img_normalized = img.astype(np.float32) / norm
        else:
            img_normalized = img.astype(np.float32)
    else:
        img_normalized = img.astype(np.float32) / 255.0
    
    return img_normalized

def enhance_contrast(img, method='clahe'):
    """
    Enhance image contrast
    
    Args:
        img: Input image (grayscale)
        method: 'clahe' (Contrast Limited Adaptive Histogram Equalization) or 'hist_eq'
    
    Returns:
        Enhanced image
    """
    if method == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img)
    elif method == 'hist_eq':
        enhanced = cv2.equalizeHist(img)
    else:
        enhanced = img
    
    return enhanced

def reduce_noise(img, method='gaussian'):
    """
    Reduce noise in image
    
    Args:
        img: Input image
        method: 'gaussian', 'bilateral', or 'median'
    
    Returns:
        Denoised image
    """
    if method == 'gaussian':
        denoised = cv2.GaussianBlur(img, (5, 5), 0)
    elif method == 'bilateral':
        denoised = cv2.bilateralFilter(img, 9, 75, 75)
    elif method == 'median':
        denoised = cv2.medianBlur(img, 5)
    else:
        denoised = img
    
    return denoised

def preprocess_image(img_path, save_preprocessed=False, output_dir=None):
    """
    Complete preprocessing pipeline for a single image
    
    Args:
        img_path: Path to input image
        save_preprocessed: Whether to save preprocessed images
        output_dir: Directory to save preprocessed images
    
    Returns:
        Dictionary with original and preprocessed image data and statistics
    """
    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    # Convert to grayscale for medical imaging analysis
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Store original statistics
    original_stats = {
        'mean': np.mean(img_gray),
        'std': np.std(img_gray),
        'min': np.min(img_gray),
        'max': np.max(img_gray),
        'shape': img_gray.shape
    }
    
    # Apply preprocessing steps
    # 1. Noise reduction
    img_denoised = reduce_noise(img_gray, method='bilateral')
    
    # 2. Contrast enhancement
    img_enhanced = enhance_contrast(img_denoised, method='clahe')
    
    # 3. Normalization
    img_normalized = normalize_image(img_enhanced, method='min_max')
    
    # Convert normalized back to uint8 for visualization
    img_normalized_uint8 = (img_normalized * 255).astype(np.uint8)
    
    # Store preprocessed statistics
    preprocessed_stats = {
        'mean': np.mean(img_normalized),
        'std': np.std(img_normalized),
        'min': np.min(img_normalized),
        'max': np.max(img_normalized),
        'shape': img_normalized.shape
    }
    
    # Save preprocessed image if requested
    if save_preprocessed and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), img_normalized_uint8)
    
    return {
        'original': img_gray,
        'preprocessed': img_normalized_uint8,
        'original_stats': original_stats,
        'preprocessed_stats': preprocessed_stats,
        'normalized': img_normalized
    }

def create_histogram(data, title, save_path=None):
    """
    Create and save histogram
    
    Args:
        data: Data to plot
        title: Plot title
        save_path: Path to save the histogram
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data.flatten(), bins=256, range=(0, 256), alpha=0.7, color='blue', edgecolor='black')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Pixel Intensity', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Histogram saved to: {save_path}")
    
    plt.close()

def copy_label_files(base_dir, split, target_dir):
    """
    Ensure the label files exist next to the preprocessed images so YOLO can
    treat the directory as a full dataset split.
    """
    source_labels = base_dir / "dataset" / split / "labels"
    if not source_labels.exists():
        print(f"Warning: label directory missing at {source_labels}")
        return

    target_labels = target_dir / "labels"
    target_labels.mkdir(parents=True, exist_ok=True)

    copied = 0
    for label_file in source_labels.glob("*.txt"):
        destination = target_labels / label_file.name
        shutil.copy2(label_file, destination)
        copied += 1

    print(f"Labels copied to {target_labels} ({copied} files)")


def analyze_dataset(base_dir, split='train', sample_size=None):
    """
    Analyze entire dataset and create histograms
    
    Args:
        base_dir: Base directory of the project
        split: Dataset split ('train', 'valid', 'test')
        sample_size: Number of images to sample (None for all)
    
    Returns:
        Dictionary with analysis results
    """
    base_dir = Path(base_dir)
    images_dir = base_dir / "dataset" / split / "images"
    
    if not images_dir.exists():
        print(f"Error: {images_dir} does not exist")
        return None
    
    # Get all image files
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    
    if sample_size:
        import random
        image_files = random.sample(image_files, min(sample_size, len(image_files)))
    
    print(f"\nAnalyzing {len(image_files)} images from {split} set...")
    
    # Storage for statistics
    all_original_pixels = []
    all_preprocessed_pixels = []
    all_stats = []
    
    # Create output directory
    output_dir = base_dir / "preprocessing_output" / split
    preprocessed_dir = output_dir / "preprocessed_images"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    for img_path in tqdm(image_files, desc=f"Processing {split} images"):
        result = preprocess_image(img_path, save_preprocessed=True, output_dir=preprocessed_dir)
        
        if result:
            all_original_pixels.extend(result['original'].flatten())
            all_preprocessed_pixels.extend(result['preprocessed'].flatten())
            all_stats.append({
                'filename': img_path.name,
                'original_mean': result['original_stats']['mean'],
                'original_std': result['original_stats']['std'],
                'original_min': result['original_stats']['min'],
                'original_max': result['original_stats']['max'],
                'preprocessed_mean': result['preprocessed_stats']['mean'],
                'preprocessed_std': result['preprocessed_stats']['std'],
                'preprocessed_min': result['preprocessed_stats']['min'],
                'preprocessed_max': result['preprocessed_stats']['max'],
            })
    
    # Convert to numpy arrays
    all_original_pixels = np.array(all_original_pixels)
    all_preprocessed_pixels = np.array(all_preprocessed_pixels)
    
    # Create histograms
    print("\nCreating histograms...")
    create_histogram(
        all_original_pixels,
        f'Pixel Intensity Distribution - {split.upper()} Set (Original)',
        output_dir / f'histogram_original_{split}.png'
    )
    
    create_histogram(
        all_preprocessed_pixels,
        f'Pixel Intensity Distribution - {split.upper()} Set (Preprocessed)',
        output_dir / f'histogram_preprocessed_{split}.png'
    )
    
    # Create comparison histogram
    plt.figure(figsize=(12, 6))
    plt.hist(all_original_pixels, bins=256, range=(0, 256), alpha=0.5, 
             label='Original', color='blue', edgecolor='black')
    plt.hist(all_preprocessed_pixels, bins=256, range=(0, 256), alpha=0.5, 
             label='Preprocessed', color='red', edgecolor='black')
    plt.title(f'Pixel Intensity Distribution Comparison - {split.upper()} Set', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Pixel Intensity', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'histogram_comparison_{split}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison histogram saved to: {output_dir / f'histogram_comparison_{split}.png'}")
    
    # Save statistics to CSV
    stats_df = pd.DataFrame(all_stats)
    stats_csv_path = output_dir / f'statistics_{split}.csv'
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"Statistics saved to: {stats_csv_path}")

    # Mirror the labels so the preprocessed folder becomes a full dataset
    copy_label_files(base_dir, split, output_dir)
    
    # Print summary statistics
    print(f"\n=== Summary Statistics for {split.upper()} Set ===")
    print(f"Original Images:")
    print(f"  Mean: {np.mean(all_original_pixels):.2f}")
    print(f"  Std: {np.std(all_original_pixels):.2f}")
    print(f"  Min: {np.min(all_original_pixels):.0f}")
    print(f"  Max: {np.max(all_original_pixels):.0f}")
    print(f"\nPreprocessed Images:")
    print(f"  Mean: {np.mean(all_preprocessed_pixels):.2f}")
    print(f"  Std: {np.std(all_preprocessed_pixels):.2f}")
    print(f"  Min: {np.min(all_preprocessed_pixels):.0f}")
    print(f"  Max: {np.max(all_preprocessed_pixels):.0f}")
    
    return {
        'original_pixels': all_original_pixels,
        'preprocessed_pixels': all_preprocessed_pixels,
        'statistics': stats_df,
        'output_dir': output_dir
    }

def main():
    """Main function to run preprocessing pipeline"""
    base_dir = Path(__file__).resolve().parent
    
    print("=" * 60)
    print("Bone Fracture Detection - Data Preprocessing Pipeline")
    print("=" * 60)
    
    # Process each dataset split
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing {split.upper()} set...")
        print(f"{'='*60}")
        
        # Analyze dataset (use sample for faster processing, set to None for full dataset)
        result = analyze_dataset(base_dir, split=split, sample_size=None)
        
        if result:
            print(f"\n✓ {split.upper()} set processing completed!")
            print(f"  Output directory: {result['output_dir']}")
    
    print(f"\n{'='*60}")
    print("Preprocessing pipeline completed successfully!")
    print(f"{'='*60}")
    print(f"\nAll outputs saved to: {base_dir / 'preprocessing_output'}")

if __name__ == "__main__":
    main()

