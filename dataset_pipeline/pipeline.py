from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import yaml
import torch
from rich.console import Console

from .catalog import load_catalog
from .frame_extraction import FrameExtractor
from .ingestion import ingest_videos
from .label_generation import LabelWriter
from .postprocessing import BallCandidateBuilder
from .qc import QualityControlPlanner
from .splitting import DatasetSplitter
from .teacher_inference import TeacherInferencer


console = Console()


def run_pipeline(config_path: str | Path) -> None:
    config = load_config(config_path)
    workspace_cfg = config["workspace"]

    raw_root = Path(workspace_cfg["raw_video_root"])
    manifest_path = Path(workspace_cfg["catalog_manifest"])
    frames_root = Path(workspace_cfg["frames_root"])
    detections_root = Path(workspace_cfg["detections_root"])
    labels_ball = Path(workspace_cfg["labels"]["ball"])
    qc_root = Path(workspace_cfg["qc_root"])
    splits_root = Path(workspace_cfg["splits_root"])
    manifests_root = Path(workspace_cfg["manifests_root"])
    visualizations_root = Path(workspace_cfg.get("visualizations_root", "artifacts/visualizations"))
    visualizations_root.mkdir(parents=True, exist_ok=True)

    console.log("[bold]1. Raw Data Ingestion[/bold]")
    ingest_videos(raw_root, manifest_path, metadata_hook=default_metadata_hook)
    catalog = load_catalog(manifest_path)
    console.log(f"Catalog contains {len(catalog)} videos.")

    console.log("[bold]2. Frame Extraction[/bold]")
    frame_index_path = manifests_root / "frames_index.csv"
    extractor = FrameExtractor(
        frames_root=frames_root,
        sampling_strategy=config["frame_extraction"]["sampling_strategy"],
        stride=config["frame_extraction"]["stride"],
        target_fps=config["frame_extraction"]["target_fps"],
        image_format=config["frame_extraction"]["image_format"],
        quality=config["frame_extraction"]["quality"],
        index_path=frame_index_path,
    )
    frame_records = extractor.extract(catalog)
    console.log(f"Extracted {len(frame_records)} frames.")

    if not frame_records:
        console.log(
            "[yellow]No frames extracted. "
            "Skipping teacher inference, labeling, and split generation.[/yellow]"
        )
        return

    console.log("[bold]3. Teacher Model Inference[/bold]")
    teacher_device = resolve_device(config["teacher_inference"]["device"])
    alias_map = config["teacher_inference"].get("alias_map", {})
    inferencer = TeacherInferencer(
        output_root=detections_root,
        alias_map=alias_map,
        model_name=config["teacher_inference"]["model_name"],
        weights_path=config["teacher_inference"]["weights_path"],
        device=teacher_device,
        confidence_threshold=config["teacher_inference"]["confidence_threshold"],
        iou_threshold=config["teacher_inference"]["iou_threshold"],
        batch_size=config["teacher_inference"]["batch_size"],
        tracker_config=config["teacher_inference"].get("tracker"),
        target_class_name=config["teacher_inference"].get("target_class_name"),
    )
    detection_paths = inferencer.run(frame_records)
    console.log(f"Saved detections for {len(detection_paths)} frames.")

    console.log("[bold]4. Ball Filtering[/bold]")
    ball_cfg = config["postprocessing"]["ball"]
    builder = BallCandidateBuilder(ball_config=ball_cfg)
    ball_records = builder.build(detection_paths)
    console.log("Ball stream:", len(ball_records), "frames")

    console.log("[bold]5. YOLO Label Generation[/bold]")
    ball_writer = LabelWriter(labels_ball, {"ball": 0})
    ball_writer.write(ball_records)

    console.log("[bold]6. Quality Control Review Lists[/bold]")
    qc = QualityControlPlanner(
        qc_root,
        config["quality_control"]["priority_rules"],
    )
    qc.build(ball_records)

    console.log("[bold]7. Dataset Splitting & Versioning[/bold]")
    splitter = DatasetSplitter(
        splits_root=splits_root,
        strategy=config["splits"]["strategy"],
        proportions=config["splits"]["proportions"],
        min_images_per_split=config["splits"]["min_images_per_split"],
    )
    ball_splits = splitter.split(ball_records, "ball")
    write_dataset_manifest(
        manifests_root / f"ball_dataset_{config['dataset_version']}.json",
        frames_root,
        labels_ball,
        ball_splits,
        {"ball": 0},
        config["dataset_version"],
    )
    console.log("[bold]8. Teacher Visualization Overlays[/bold]")
    label_text = config["teacher_inference"].get("visualization_label", "ball")
    for video in catalog:
        output_path = (
            visualizations_root
            / video.session_id
            / f"{video.camera_id}_{Path(video.video_path).stem}_teacher.mp4"
        )
        try:
            inferencer.render_video(video, output_path, label=label_text)
        except Exception as exc:  # pragma: no cover - best-effort visualization
            console.log(f"[yellow]Failed to render {video.video_path}: {exc}[/yellow]")

    console.log("[green]Part 1 pipeline completed.[/green]")


def default_metadata_hook(path: Path) -> Dict[str, str]:
    stats = path.stat()
    return {
        "filesize": stats.st_size,
        "modified_at": stats.st_mtime,
    }


def load_config(path: str | Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def write_dataset_manifest(
    manifest_path: Path,
    images_root: Path,
    labels_root: Path,
    splits: Dict[str, List[str]],
    class_map: Dict[str, int],
    version: str,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    splits_relative = {
        split: [relativize(Path(path), images_root) for path in paths]
        for split, paths in splits.items()
    }
    payload = {
        "version": version,
        "images_root": str(images_root),
        "labels_root": str(labels_root),
        "splits": splits_relative,
        "classes": class_map,
    }
    manifest_path.write_text(json.dumps(payload, indent=2))


def relativize(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def resolve_device(preferred: str) -> str:
    normalized = preferred.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        console.log(
            "[yellow]CUDA requested but not available. Falling back to CPU.[/yellow]"
        )
        return "cpu"
    if normalized == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if not (mps_backend and torch.backends.mps.is_available()):
            console.log(
                "[yellow]MPS requested but not available. Falling back to CPU.[/yellow]"
            )
            return "cpu"
    return preferred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 1 dataset pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
