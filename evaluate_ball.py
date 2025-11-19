from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

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
    ensure_ultralytics_available()
    config = load_training_config(args.config)

    experiment_cfg = config["experiment"]
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]

    exp_root = Path(experiment_cfg.get("output_dir", "artifacts/training")) / experiment_cfg["name"]
    manifest = DatasetManifest.load(dataset_cfg["manifest_path"])
    preparer = YOLODatasetPreparer(
        manifest=manifest,
        staging_root=dataset_cfg.get("staging_root", exp_root / "dataset"),
        experiment_name=experiment_cfg["name"],
        console=console,
    )
    dataset_yaml = preparer.prepare()

    weights = Path(args.weights) if args.weights else Path(model_cfg["base_weights"])
    console.log(f"Evaluating weights: {weights}")
    device = resolve_device(model_cfg.get("device", "auto"), console)

    model = YOLO(str(weights))
    eval_kwargs = build_eval_kwargs(config, dataset_yaml, exp_root, device)
    model.val(**eval_kwargs)


def build_eval_kwargs(
    config: Dict[str, Any],
    dataset_yaml: Path,
    exp_root: Path,
    device: str,
) -> Dict[str, Any]:
    eval_cfg = config.get("evaluation", {})
    kwargs: Dict[str, Any] = {
        "data": str(dataset_yaml),
        "split": eval_cfg.get("split", "test"),
        "project": str(exp_root),
        "name": eval_cfg.get("name", "test"),
        "exist_ok": True,
        "device": device,
    }
    training_cfg = config.get("training", {})
    if "batch" in training_cfg:
        kwargs["batch"] = training_cfg["batch"]
    if "imgsz" in training_cfg:
        kwargs["imgsz"] = training_cfg["imgsz"]
    kwargs.update(eval_cfg.get("overrides", {}))
    return kwargs


def ensure_ultralytics_available() -> None:
    if YOLO is None:
        raise ImportError(
            "ultralytics is required for evaluation. Install it via `pip install ultralytics`."
        ) from _import_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a lightweight YOLO ball detector.")
    parser.add_argument("--config", required=True, help="Training config path.")
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to the checkpoint to evaluate. Defaults to the base weights from the config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
