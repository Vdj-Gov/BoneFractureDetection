# Bone Fracture Detection

A deep learning project for detecting bone fractures using YOLOv8.

## Project Structure

- `Step01-Preprocess.py` - Data preprocessing script
- `Step03-Train.py` - Model training script
- `Step04-Predict.py` - Prediction/inference script
- `Step05-ConfusionMatrix.py` - Confusion matrix generation
- `Step06-Ablation.py` - Ablation study script

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

## Usage

1. **Preprocess data:**
   ```bash
   python Step01-Preprocess.py
   ```

2. **Train model:**
   ```bash
   python Step03-Train.py
   ```

3. **Make predictions:**
   ```bash
   python Step04-Predict.py
   ```

4. **Generate confusion matrix:**
   ```bash
   python Step05-ConfusionMatrix.py
   ```

5. **Run ablation study:**
   ```bash
   python Step06-Ablation.py
   ```

## Configuration

- `data.yaml` - Dataset configuration
- `data_preprocessed.yaml` - Preprocessed dataset configuration
- `ablation_config.yaml` - Ablation study configuration

## Notes

- Model weights (`.pt` files) are excluded from git due to size
- Dataset files should be placed in the `dataset/` directory
- Training outputs are saved in the `runs/` directory

