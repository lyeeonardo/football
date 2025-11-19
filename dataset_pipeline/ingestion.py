from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

from .catalog import VideoRecord, load_catalog, save_catalog

MetadataHook = Callable[[Path], Dict[str, Any]]


def ingest_videos(
    raw_root: str | Path,
    manifest_path: str | Path,
    metadata_hook: MetadataHook | None = None,
) -> List[VideoRecord]:
    """Traverse the raw video directory and build/extend the catalog manifest."""

    raw_root = Path(raw_root)
    manifest_path = Path(manifest_path)
    metadata_hook = metadata_hook or (lambda _: {})

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw video root not found: {raw_root}")

    current_catalog = load_catalog(manifest_path)
    existing = {record.video_path for record in current_catalog}
    catalog: List[VideoRecord] = []

    flat_videos = sorted(raw_root.glob("*.mp4"))
    if flat_videos:
        catalog.extend(
            _records_from_flat_layout(flat_videos, existing, metadata_hook)
        )
    else:
        catalog.extend(
            _records_from_nested_layout(raw_root, existing, metadata_hook)
        )

    if catalog:
        save_catalog([*current_catalog, *catalog], manifest_path)

    return catalog


def sha256(path: Path) -> str:
    """Helpful hash for integrity tracking."""

    sha = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _records_from_flat_layout(
    videos: List[Path],
    existing: set[str],
    metadata_hook: MetadataHook,
) -> List[VideoRecord]:
    records: List[VideoRecord] = []
    for idx, video_file in enumerate(videos):
        if str(video_file) in existing:
            continue
        metadata = metadata_hook(video_file)
        session_id = str(metadata.pop("session_id", video_file.stem))
        camera_id = str(metadata.pop("camera_id", "default_camera"))
        record = VideoRecord(
            session_id=session_id,
            camera_id=camera_id,
            video_path=str(video_file),
            metadata={
                "hash": sha256(video_file),
                **metadata,
            },
        )
        records.append(record)
        existing.add(str(video_file))
    return records


def _records_from_nested_layout(
    raw_root: Path,
    existing: set[str],
    metadata_hook: MetadataHook,
) -> List[VideoRecord]:
    records: List[VideoRecord] = []
    for session_dir in sorted(d for d in raw_root.iterdir() if d.is_dir()):
        session_id = session_dir.name
        for camera_dir in sorted(d for d in session_dir.iterdir() if d.is_dir()):
            camera_id = camera_dir.name
            for video_file in sorted(camera_dir.glob("*.mp4")):
                if str(video_file) in existing:
                    continue
                metadata = metadata_hook(video_file)
                record = VideoRecord(
                    session_id=session_id,
                    camera_id=camera_id,
                    video_path=str(video_file),
                    metadata={
                        "hash": sha256(video_file),
                        **metadata,
                    },
                )
                records.append(record)
                existing.add(str(video_file))
    return records
