# Bone Fracture Detection

A comprehensive deep learning project for automated bone fracture detection in medical X-ray images using YOLOv8 (You Only Look Once version 8) object detection model.

## 📋 Description

This project implements an end-to-end machine learning pipeline for detecting bone fractures in X-ray images. The system uses YOLOv8, a state-of-the-art object detection architecture, to identify and localize fractures in medical images. The project includes data preprocessing, model training, inference, evaluation, and comprehensive ablation studies to optimize model performance.

### Key Features

- **Automated Preprocessing Pipeline**: Normalizes images, enhances contrast (CLAHE), reduces noise, and generates statistical analysis
- **YOLOv8 Object Detection**: Uses YOLOv8 Large model for accurate fracture detection and localization
- **Comprehensive Evaluation**: Generates confusion matrices, classification reports, and performance metrics
- **Ablation Studies**: Systematic experiments to evaluate different preprocessing techniques and model configurations
- **Visualization Tools**: Creates histograms, comparison plots, and prediction visualizations
- **PyTorch 2.6+ Compatibility**: Includes fixes for latest PyTorch versions

## 🔬 Project Workflow

The project follows a structured 6-step pipeline:

1. **Step 01 - Preprocessing**: Prepares and enhances X-ray images for training
2. **Step 03 - Training**: Trains the YOLOv8 model on preprocessed data
3. **Step 04 - Prediction**: Performs inference on new images
4. **Step 05 - Evaluation**: Generates confusion matrices and performance metrics
5. **Step 06 - Ablation Study**: Systematic experiments to optimize model performance

## 📁 Project Structure

- `Step01-Preprocess.py` - Data preprocessing script with normalization, contrast enhancement, and noise reduction
- `Step03-Train.py` - Model training script using YOLOv8 Large architecture
- `Step04-Predict.py` - Prediction/inference script with visualization capabilities
- `Step05-ConfusionMatrix.py` - Model evaluation and confusion matrix generation
- `Step06-Ablation.py` - Automated ablation study script for hyperparameter optimization
- `data.yaml` - Dataset configuration file
- `data_preprocessed.yaml` - Preprocessed dataset configuration
- `ablation_config.yaml` - Ablation study experiment configurations
- `req.txt` - Installation requirements and setup instructions

## Setup

### Prerequisites

- Python 3.8
- CUDA 11.8 (for GPU support)
- Conda

### Installation

1. Create conda environment:
```bash
conda create --name YoloV8 python=3.8
conda activate YoloV8
```

2. Check CUDA version:
```bash
nvcc --version
```

3. Install PyTorch with CUDA 11.8:
```bash
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

4. Install YOLOv8:
```bash
pip install ultralytics==8.1.0
pip install lapx>=0.5.2
```

## 🚀 Usage

### Step 1: Data Preprocessing
Preprocesses X-ray images with normalization, contrast enhancement (CLAHE), and noise reduction:
```bash
python Step01-Preprocess.py
```
**Output**: Preprocessed images, histograms, and statistical analysis saved in `preprocessing_output/`

### Step 2: Model Training
Trains YOLOv8 Large model on the preprocessed dataset:
```bash
python Step03-Train.py
```
**Configuration**:
- Model: YOLOv8 Large (`yolov8l.pt`)
- Epochs: 1000 (with early stopping patience: 300)
- Batch size: 16
- Image size: 350x350
- Device: GPU (CUDA)

**Output**: Trained model weights and training metrics saved in `runs/train/`

### Step 3: Make Predictions
Performs inference on new X-ray images:
```bash
python Step04-Predict.py
```
**Output**: Prediction visualizations with bounding boxes saved in `output/`

### Step 4: Model Evaluation
Generates confusion matrix and classification metrics:
```bash
python Step05-ConfusionMatrix.py
```
**Output**: Confusion matrices (normalized and raw) saved in `evaluation_output/`

### Step 5: Ablation Study
Runs systematic experiments to evaluate different configurations:
```bash
python Step06-Ablation.py
```
**Options**:
- `--only <experiment_names>`: Run specific experiments
- `--skip-existing`: Skip already completed runs
- `--dry-run`: Preview planned experiments without executing

**Output**: Ablation results and comparisons saved in `evaluation_output/`

## ⚙️ Configuration Files

- **`data.yaml`**: Defines dataset paths and class names for original data
  - Train/validation/test splits
  - Class definitions (Fracture detection)
  
- **`data_preprocessed.yaml`**: Configuration for preprocessed dataset paths

- **`ablation_config.yaml`**: Defines ablation study experiments
  - Different preprocessing techniques
  - Model variants
  - Hyperparameter combinations

## 📊 Dataset Structure

The project expects the following dataset structure:
```
dataset/
├── train/
│   ├── images/     # Training X-ray images
│   └── labels/     # YOLO format annotations (.txt)
├── valid/
│   ├── images/     # Validation images
│   └── labels/     # Validation annotations
└── test/
    ├── images/     # Test images
    └── labels/     # Test annotations
```

## 🔧 Technical Details

### Preprocessing Techniques
- **Normalization**: Min-max, Z-score, and unit vector normalization
- **Contrast Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization) and histogram equalization
- **Noise Reduction**: Gaussian blur, bilateral filter, and median filter

### Model Architecture
- **Base Model**: YOLOv8 Large (YOLOv8l)
- **Input Size**: 350x350 pixels
- **Task**: Object Detection (single class: Fracture)
- **Framework**: Ultralytics YOLOv8

### Training Configuration
- **Optimizer**: Adam (default YOLOv8 optimizer)
- **Loss Function**: Combined classification and localization loss
- **Early Stopping**: Patience of 300 epochs
- **Validation**: Automatic validation during training

## 📝 Notes

- **Model Weights**: Pre-trained and trained model weights (`.pt` files) are excluded from git due to size constraints
- **Dataset**: Original dataset files are excluded. Place your dataset in the `dataset/` directory
- **Outputs**: Training outputs, predictions, and evaluation results are saved in respective directories
- **Cache Files**: Label cache files (`.cache`) are automatically generated and excluded from version control

## 🛠️ Dependencies

- Python 3.8
- PyTorch 2.1.1 (with CUDA 11.8)
- Ultralytics YOLOv8 8.1.0
- OpenCV (cv2)
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- PyYAML
- pandas
- tqdm

## 📄 License

This project is for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on the GitHub repository.

