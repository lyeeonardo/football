from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import cv2

from .catalog import FrameRecord, VideoRecord


try:  # pragma: no cover - heavy dependency
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
    YOLO = None  # type: ignore
    _import_error = exc
else:
    _import_error = None


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox_xyxy: List[float]

    def to_dict(self) -> Dict[str, float | str | List[float]]:
        return asdict(self)


class TeacherInferencer:
    """Runs YOLOv8x6 teacher model to generate pseudo labels for every frame."""

    def __init__(
        self,
        output_root: str | Path,
        alias_map: Dict[str, str] | None = None,
        model_name: str = "yolov8x6",
        weights_path: str | None = None,
        device: str = "cuda",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        batch_size: int = 32,
        metadata_path: str | Path | None = None,
        tracker_config: str | None = None,
        target_class_name: str | None = None,
    ) -> None:
        if YOLO is None:
            raise ImportError(
                "ultralytics is not available. Install it to run teacher inference."
            ) from _import_error

        weights = weights_path or model_name
        self.model = YOLO(weights)
        self.output_root = Path(output_root)
        self.alias_map = alias_map or {}
        self.model_names = self._load_model_names()
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.batch_size = batch_size
        self.metadata_path = Path(metadata_path) if metadata_path else (
            self.output_root / "inference_metadata.json"
        )
        self.tracker_config = tracker_config
        self.target_class_ids = self._resolve_target_class_ids(target_class_name)

    def run(self, frames: Iterable[FrameRecord]) -> List[Path]:
        frame_list = list(frames)
        saved: List[Path] = []
        for chunk in chunked(frame_list, self.batch_size):
            if not chunk:
                continue
            image_paths = [frame.frame_path for frame in chunk]
            results = self.model.predict(
                image_paths,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )
            for frame, result in zip(chunk, results):
                detections = self._extract_detections(result)
                json_path = self._json_path(frame)
                json_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "frame": frame.to_dict(),
                    "detections": [d.to_dict() for d in detections],
                }
                json_path.write_text(json.dumps(payload, indent=2))
                saved.append(json_path)
        self._write_metadata(len(frame_list))
        return saved

    def _extract_detections(self, result) -> List[Detection]:  # pragma: no cover - YOLO object
        detections: List[Detection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections
        cls_ids = boxes.cls.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        xyxy = boxes.xyxy.cpu().tolist()
        for class_id, conf, bbox in zip(cls_ids, confidences, xyxy):
            if conf < self.confidence_threshold:
                continue
            if self.target_class_ids and int(class_id) not in self.target_class_ids:
                continue
            raw_name = self._class_name_from_id(int(class_id))
            class_name = self.alias_map.get(raw_name, raw_name)
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=float(conf),
                    bbox_xyxy=[float(x) for x in bbox],
                )
            )
        return detections

    def _json_path(self, frame: FrameRecord) -> Path:
        filename = Path(frame.frame_path).stem + ".json"
        return self.output_root / frame.session_id / frame.camera_id / filename

    def _write_metadata(self, frame_count: int) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        metadata = {
            "model": str(self.model.model if hasattr(self.model, "model") else "unknown"),
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "model_class_names": self.model_names,
            "alias_map": self.alias_map,
            "frames_processed": frame_count,
        }
        self.metadata_path.write_text(json.dumps(metadata, indent=2))

    def _load_model_names(self) -> Dict[int, str]:
        names_attr = getattr(self.model, "names", {})
        if isinstance(names_attr, dict):
            return {int(k): v for k, v in names_attr.items()}
        if isinstance(names_attr, list):
            return {idx: name for idx, name in enumerate(names_attr)}
        return {}

    def _class_name_from_id(self, class_id: int) -> str:
        return self.model_names.get(class_id, str(class_id))

    def _resolve_target_class_ids(self, target_name: str | None) -> set[int] | None:
        if not target_name:
            return None
        normalized = target_name.lower()
        ids = {idx for idx, name in self.model_names.items() if name.lower() == normalized}
        return ids or None

    # Visualization helpers
    def render_video(self, video: VideoRecord, output_path: Path, label: str = "ball") -> None:
        source_path = Path(video.video_path)
        if not source_path.exists():
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fps, width, height = self._read_video_metadata(source_path)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        stream = self.model.track(
            source=str(source_path),
            tracker=self.tracker_config,
            stream=True,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            persist=True,
            verbose=False,
        )
        try:
            for result in stream:
                frame, _ = self._annotate_tracking_result(result, label)
                writer.write(frame)
        finally:
            writer.release()

    def _read_video_metadata(self, path: Path) -> Tuple[float, int, int]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()
        return fps, width, height

    def _annotate_tracking_result(self, result, label: str) -> Tuple:
        frame = result.orig_img.copy()
        detected = False
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            ids = getattr(boxes, "id", None)
            for idx, (xyxy, conf, cls) in enumerate(zip(boxes.xyxy, boxes.conf, boxes.cls)):
                class_id = int(cls)
                if self.target_class_ids and class_id not in self.target_class_ids:
                    continue
                detected = True
                x1, y1, x2, y2 = [int(float(v)) for v in xyxy]
                text = f"{label} {float(conf):.2f}"
                if ids is not None and len(ids) > idx and ids[idx] is not None:
                    text += f" ID {int(ids[idx])}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(
                    frame,
                    text,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )
        if not detected:
            cv2.putText(
                frame,
                f"{label}: no detection",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        return frame, detected


def chunked(seq: Sequence[FrameRecord], size: int) -> Iterator[List[FrameRecord]]:
    for idx in range(0, len(seq), size):
        yield list(seq[idx : idx + size])
