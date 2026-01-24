import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except Exception as e:
    print("Error importing ultralytics:", e)
    print("Install requirements with: python -m pip install -r req.txt")
    sys.exit(1)


def main():
    base_dir = Path(__file__).resolve().parent

    # locate data.yaml next to this script
    config_file = base_dir / "data.yaml"
    if not config_file.exists():
        print(f"data.yaml not found at {config_file}")
        print("Make sure data.yaml is placed next to Step03-Train.py or update the path.")
        sys.exit(1)

    # choose a model. Use a released weights file name so ultralytics can download it if missing.
    model_name = "yolov8l.pt"

    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"Failed to load model '{model_name}':", e)
        print("You can try a smaller model like 'yolov8n.pt' if you have limited resources.")
        sys.exit(1)

    # safer defaults
    project = str(base_dir / "runs" / "train")
    experiment = "My-Model"
    batch_size = 16

    # start training
    try:
        result = model.train(
            data=str(config_file),
            epochs=1000,
            project=project,
            name=experiment,
            batch=batch_size,
            device=0,
            patience=300,
            imgsz=350,
            verbose=True,
            val=True,
        )
        print("Training started. Results saved to:", project)
    except Exception as e:
        print("Training failed:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()