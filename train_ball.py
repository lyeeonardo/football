from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from rich.console import Console

from training.config import load_training_config
from training.dataset import DatasetManifest, YOLODatasetPreparer
from training.utils import resolve_device

try:  # pragma: no cover - heavy dependency
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
    YOLO = None  # type: ignore
    _import_error = exc
else:
    _import_error = None


console = Console()


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    ensure_ultralytics_available()

    experiment_cfg = config["experiment"]
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]

    exp_root = Path(experiment_cfg.get("output_dir", "artifacts/training")) / experiment_cfg["name"]
    exp_root.mkdir(parents=True, exist_ok=True)

    console.log("[bold]Preparing dataset staging area[/bold]")
    manifest = DatasetManifest.load(dataset_cfg["manifest_path"])
    preparer = YOLODatasetPreparer(
        manifest=manifest,
        staging_root=dataset_cfg.get("staging_root", exp_root / "dataset"),
        experiment_name=experiment_cfg["name"],
        console=console,
    )
    dataset_yaml = preparer.prepare()
    console.log(f"Dataset YAML: {dataset_yaml}")

    device = resolve_device(model_cfg.get("device", "auto"), console)
    console.log(f"Using device: {device}")

    seed = int(experiment_cfg.get("seed", 42))
    set_random_seeds(seed)

    console.log("[bold]Initializing YOLO model[/bold]")
    model = YOLO(model_cfg["base_weights"])

    train_kwargs = build_train_kwargs(config, dataset_yaml, exp_root, device, seed)
    console.log("[bold]Starting training[/bold]")
    trainer = model.train(**train_kwargs)
    console.log(f"[green]Training complete. Artifacts saved to {trainer.save_dir}[/green]")

    best_weights = Path(trainer.save_dir) / "weights" / "best.pt"
    if best_weights.exists():
        console.log(f"Best checkpoint: {best_weights}")

    if "test" in manifest.split_names():
        console.log("[bold]Evaluating on held-out test split[/bold]")
        eval_kwargs = build_eval_kwargs(config, dataset_yaml, exp_root, device, seed)
        model.val(**eval_kwargs)


def build_train_kwargs(
    config: Dict[str, Any],
    dataset_yaml: Path,
    exp_root: Path,
    device: str,
    seed: int,
) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    overrides.update(config.get("training", {}))
    overrides.update(config.get("augmentation", {}))
    overrides.setdefault("imgsz", 640)
    overrides.setdefault("epochs", 50)
    overrides.setdefault("batch", 16)
    overrides["data"] = str(dataset_yaml)
    overrides["device"] = device
    overrides["project"] = str(exp_root)
    overrides["name"] = "train"
    overrides["exist_ok"] = True
    overrides["seed"] = seed
    return overrides


def build_eval_kwargs(
    config: Dict[str, Any],
    dataset_yaml: Path,
    exp_root: Path,
    device: str,
    seed: int,
) -> Dict[str, Any]:
    eval_cfg = config.get("evaluation", {})
    kwargs: Dict[str, Any] = {
        "data": str(dataset_yaml),
        "split": eval_cfg.get("split", "test"),
        "device": device,
        "project": str(exp_root),
        "name": eval_cfg.get("name", "test"),
        "exist_ok": True,
        "seed": seed,
    }
    if "batch" in config.get("training", {}):
        kwargs["batch"] = config["training"]["batch"]
    if "imgsz" in config.get("training", {}):
        kwargs["imgsz"] = config["training"]["imgsz"]
    kwargs.update(eval_cfg.get("overrides", {}))
    return kwargs


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_ultralytics_available() -> None:
    if YOLO is None:
        raise ImportError(
            "ultralytics is required for training. Install it via `pip install ultralytics`."
        ) from _import_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a lightweight YOLO ball detector.")
    parser.add_argument("--config", required=True, help="Path to training YAML config.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
