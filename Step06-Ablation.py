
import argparse
import csv
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

try:
    import yaml
except ImportError as exc:  
    print("Missing dependency 'pyyaml'. Install it with: pip install pyyaml")
    raise

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASE_DIR / "ablation_config.yaml"
SUMMARY_DIR = BASE_DIR / "evaluation_output"


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv8 ablation experiments.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Path to the ablation_config.yaml file.")
    parser.add_argument("--only", nargs="*", help="Experiment names to run.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip experiments that already have results.csv.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned runs without executing training.")
    return parser.parse_args()


def load_config(cfg_path: Path) -> Dict:
    if not cfg_path.exists():
        raise FileNotFoundError(f"Ablation config not found at {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_model_path(model_name: str) -> str:
    """Try a few sensible locations before giving the raw name to YOLO."""
    candidates = [
        BASE_DIR / model_name,
        BASE_DIR.parent / model_name,
        Path(model_name),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return model_name  


def resolve_data_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
 
    candidate = BASE_DIR / value
    if candidate.exists():
        return str(candidate)
    return str(candidate)  


def read_results_csv(csv_path: Path) -> Optional[Dict[str, str]]:
    if not csv_path.exists():
        return None
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        last_row = None
        for row in reader:
            last_row = row
    return last_row


def build_train_kwargs(global_cfg: Dict, exp_cfg: Dict) -> Dict:
    defaults = {
        "epochs": global_cfg.get("epochs", 200),
        "batch": global_cfg.get("batch", 16),
        "imgsz": global_cfg.get("imgsz", 352),
        "patience": global_cfg.get("patience", 50),
        "device": global_cfg.get("device", 0),
        "seed": global_cfg.get("seed", 42),
        "save": global_cfg.get("save", True),
        "exist_ok": global_cfg.get("exist_ok", True),
        "data": str(BASE_DIR / global_cfg.get("data", "data.yaml")),
    }
    overrides = exp_cfg.get("params", {})
    merged = {**defaults, **overrides}

    # Normalize key paths
    if "data" in merged:
        merged["data"] = resolve_data_path(merged["data"])

    return merged


def write_summary(records: List[Dict], csv_path: Path, md_path: Path):
    if not records:
        return

    SUMMARY_DIR.mkdir(exist_ok=True)
    fieldnames = [
        "experiment", "description", "model", "data", "imgsz", "epochs",
        "batch", "precision", "recall", "map50", "map50_95", "weights_path"
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    lines = [
        f"# Ablation Summary ({datetime.utcnow().isoformat(timespec='seconds')} UTC)",
        "",
        "| Experiment | Model | Data | mAP50 | Precision | Recall | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        lines.append(
            f"| {record['experiment']} | {Path(record['model']).name} | "
            f"{Path(record['data']).name} | {record['map50'] or 'n/a'} | "
            f"{record['precision'] or 'n/a'} | {record['recall'] or 'n/a'} | "
            f"{record['description']} |"
        )

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main():
    args = parse_args()
    config = load_config(args.config)
    global_cfg = config.get("global", {})
    experiments = config.get("experiments", [])

    if not experiments:
        print("No experiments defined in the config file.")
        return

    project_dir = BASE_DIR / global_cfg.get("project_dir", "runs/ablation")
    project_dir.mkdir(parents=True, exist_ok=True)

    summary_records: List[Dict] = []

    for exp in experiments:
        if not exp.get("enabled", True):
            continue

        name = exp.get("name")
        if not name:
            print("Skipping experiment without a name entry.")
            continue

        if args.only and name not in args.only:
            continue

        description = exp.get("description", "").strip()
        train_kwargs = build_train_kwargs(global_cfg, exp)
        model_name = exp.get("model", global_cfg.get("model", "yolov8m.pt"))
        model_path = resolve_model_path(model_name)

        experiment_dir = project_dir / name
        results_csv_path = experiment_dir / "results.csv"
        weights_path = experiment_dir / "weights" / "best.pt"

        print("\n" + "=" * 80)
        print(f"Experiment: {name}")
        print(f"Description: {description or 'n/a'}")
        print(f"Model checkpoint: {model_path}")
        print(f"Train params: {train_kwargs}")
        print("=" * 80 + "\n")

        already_done = results_csv_path.exists()
        if args.dry_run:
            status = "SKIPPED (dry-run mode)"
        elif args.skip_existing and already_done:
            status = "SKIPPED (results already exist)"
        else:
            try:
                model = YOLO(model_path)
            except Exception as exc:
                print(f"Failed to load model '{model_path}': {exc}")
                continue

            try:
                model.train(project=str(project_dir), name=name, **train_kwargs)
                status = "COMPLETED"
            except Exception as exc:
                print(f"Training failed for experiment '{name}': {exc}")
                status = "FAILED"

        metrics_row = read_results_csv(results_csv_path)
        record = {
            "experiment": name,
            "description": description,
            "model": model_path,
            "data": train_kwargs.get("data"),
            "imgsz": train_kwargs.get("imgsz"),
            "epochs": train_kwargs.get("epochs"),
            "batch": train_kwargs.get("batch"),
            "precision": metrics_row.get("metrics/precision(B)") if metrics_row else None,
            "recall": metrics_row.get("metrics/recall(B)") if metrics_row else None,
            "map50": metrics_row.get("metrics/mAP50(B)") if metrics_row else None,
            "map50_95": metrics_row.get("metrics/mAP50-95(B)") if metrics_row else None,
            "weights_path": str(weights_path if weights_path.exists() else ""),
        }
        summary_records.append(record)

        print(f"Status: {status}")
        if metrics_row:
            print(f"  mAP50: {metrics_row.get('metrics/mAP50(B)')}")
            print(f"  mAP50-95: {metrics_row.get('metrics/mAP50-95(B)')}")
            print(f"  Precision: {metrics_row.get('metrics/precision(B)')}")
            print(f"  Recall: {metrics_row.get('metrics/recall(B)')}")
        else:
            print("  Metrics unavailable (results.csv missing).")

    if summary_records:
        csv_path = SUMMARY_DIR / "ablation_results.csv"
        md_path = SUMMARY_DIR / "ablation_results.md"
        write_summary(summary_records, csv_path, md_path)
        print(f"\nSummary saved to: {csv_path}")
        print(f"Markdown summary saved to: {md_path}")
    else:
        print("No experiments were run or summarized.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Fatal error during ablation study: {exc}")
        sys.exit(1)

