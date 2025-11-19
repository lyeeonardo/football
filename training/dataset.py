from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import yaml
from rich.console import Console


@dataclass
class DatasetManifest:
    version: str
    images_root: Path
    labels_root: Path
    splits: Dict[str, List[str]]
    classes: Dict[str, int]
    manifest_path: Path

    @classmethod
    def load(cls, manifest_path: str | Path) -> "DatasetManifest":
        path = Path(manifest_path)
        raw = json.loads(path.read_text())
        images_root = _resolve_path(raw["images_root"])
        labels_root = _resolve_path(raw["labels_root"])
        return cls(
            version=raw.get("version", "unknown"),
            images_root=images_root,
            labels_root=labels_root,
            splits=raw["splits"],
            classes=raw["classes"],
            manifest_path=path,
        )

    def split_names(self) -> Sequence[str]:
        return list(self.splits.keys())


class YOLODatasetPreparer:
    """Creates an Ultralytics-friendly dataset layout with symlinks to the source assets."""

    def __init__(
        self,
        manifest: DatasetManifest,
        staging_root: str | Path,
        experiment_name: str,
        console: Console | None = None,
    ) -> None:
        self.manifest = manifest
        self.staging_root = Path(staging_root)
        self.experiment_name = experiment_name
        self.console = console or Console()

    def prepare(self) -> Path:
        dataset_root = self.staging_root / self.manifest.version / self.experiment_name
        if dataset_root.exists():
            shutil.rmtree(dataset_root)
        for split_name, rel_paths in self.manifest.splits.items():
            images_split = dataset_root / "images" / split_name
            labels_split = dataset_root / "labels" / split_name
            images_split.mkdir(parents=True, exist_ok=True)
            labels_split.mkdir(parents=True, exist_ok=True)
            for rel_path in rel_paths:
                self._link_image_and_label(rel_path, images_split, labels_split)
        dataset_yaml = dataset_root / "dataset.yaml"
        dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
        dataset_yaml.write_text(self._dataset_yaml_payload(dataset_root))
        return dataset_yaml

    def _link_image_and_label(
        self,
        relative_image_path: str,
        images_split_root: Path,
        labels_split_root: Path,
    ) -> None:
        src_image = self.manifest.images_root / relative_image_path
        if not src_image.exists():
            self.console.log(f"[yellow]Missing image:[/yellow] {src_image}")
            return
        dest_image = images_split_root / relative_image_path
        self._symlink_or_copy(src_image, dest_image)

        relative_label_path = Path(relative_image_path).with_suffix(".txt")
        src_label = self.manifest.labels_root / relative_label_path
        dest_label = labels_split_root / relative_label_path
        if src_label.exists():
            self._symlink_or_copy(src_label, dest_label)
        else:
            dest_label.parent.mkdir(parents=True, exist_ok=True)
            dest_label.write_text("")
            self.console.log(f"[yellow]Created empty label:[/yellow] {dest_label}")

    def _symlink_or_copy(self, src: Path, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            return
        try:
            os.symlink(src, dest)
        except OSError:
            shutil.copy2(src, dest)

    def _dataset_yaml_payload(self, dataset_root: Path) -> str:
        class_names = _ordered_class_names(self.manifest.classes)
        payload = {
            "path": str(dataset_root),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": class_names,
        }
        return yaml.safe_dump(payload, sort_keys=False)


def _ordered_class_names(class_map: Dict[str, int]) -> List[str]:
    if not class_map:
        return ["ball"]
    max_id = max(class_map.values())
    names = [""] * (max_id + 1)
    for name, idx in class_map.items():
        names[idx] = name
    return names


def _resolve_path(candidate: str) -> Path:
    path = Path(candidate)
    if path.is_absolute():
        return path
    return path.resolve()
