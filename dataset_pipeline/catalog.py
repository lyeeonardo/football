from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


@dataclass
class VideoRecord:
    """Represents a single raw video captured by one camera."""

    session_id: str
    camera_id: str
    video_path: str
    fps: float | None = None
    timestamp: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["video_path"] = str(self.video_path)
        return payload


@dataclass
class FrameRecord:
    """Lightweight handle that connects a sampled frame to its origin."""

    session_id: str
    camera_id: str
    video_path: str
    frame_path: str
    frame_index: int
    timestamp_sec: float | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_catalog(path: str | Path) -> List[VideoRecord]:
    catalog_path = Path(path)
    if not catalog_path.exists():
        return []
    data = yaml.safe_load(catalog_path.read_text()) or []
    return [VideoRecord(**item) for item in data]


def save_catalog(records: Iterable[VideoRecord], path: str | Path) -> None:
    catalog_path = Path(path)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = [record.to_dict() for record in records]
    catalog_path.write_text(yaml.safe_dump(serializable, sort_keys=False))


def append_to_catalog(record: VideoRecord, path: str | Path) -> None:
    catalog = load_catalog(path)
    catalog.append(record)
    save_catalog(catalog, path)
