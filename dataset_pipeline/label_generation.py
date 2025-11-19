from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import cv2

from .postprocessing import LabelCandidate
from .teacher_inference import Detection


class LabelWriter:
    """Converts label candidates into YOLO TXT files."""

    def __init__(self, labels_root: str | Path, class_map: Dict[str, int]) -> None:
        self.labels_root = Path(labels_root)
        self.class_map = class_map

    def write(self, records: Iterable[LabelCandidate]) -> List[Path]:
        paths: List[Path] = []
        for record in records:
            label_path = self._label_path(record.frame.frame_path)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            lines = self._record_to_lines(record)
            content = "\n".join(lines).strip()
            label_path.write_text(content + ("\n" if content else ""))
            paths.append(label_path)
        return paths

    def _record_to_lines(self, record: LabelCandidate) -> List[str]:
        if not record.detections:
            return []
        image = cv2.imread(record.frame.frame_path)
        if image is None:
            return []
        height, width = image.shape[:2]
        lines = []
        for detection in record.detections:
            class_id = self.class_map.get(detection.class_name)
            if class_id is None:
                continue
            line = detection_to_yolo(detection, width, height, class_id)
            if line:
                lines.append(line)
        return lines

    def _label_path(self, frame_path: str) -> Path:
        frame_rel = Path(frame_path)
        filename = frame_rel.stem + ".txt"
        camera_dir = frame_rel.parent.name
        session_dir = frame_rel.parent.parent.name if frame_rel.parent.parent else ""
        return self.labels_root / session_dir / camera_dir / filename


def detection_to_yolo(detection: Detection, width: int, height: int, class_id: int) -> str | None:
    x1, y1, x2, y2 = detection.bbox_xyxy
    x1 = max(0.0, min(float(x1), width))
    x2 = max(0.0, min(float(x2), width))
    y1 = max(0.0, min(float(y1), height))
    y2 = max(0.0, min(float(y2), height))
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    x_center = x1 + w / 2
    y_center = y1 + h / 2
    x_center /= width
    y_center /= height
    w /= width
    h /= height
    return f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
