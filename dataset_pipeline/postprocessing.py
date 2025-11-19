from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import numpy as np

from .catalog import FrameRecord
from .teacher_inference import Detection


@dataclass
class LabelCandidate:
    frame: FrameRecord
    detections: List[Detection]


class BallCandidateBuilder:
    """Filters teacher detections to produce ball label candidates."""

    def __init__(
        self,
        ball_config: Dict[str, float | bool | List[float]],
    ) -> None:
        self.ball_config = ball_config
        self.include_background_frames = bool(
            ball_config.get("include_background_frames", True)
        )

    def build(self, detection_files: Iterable[str | Path]) -> List[LabelCandidate]:
        ball_records: List[LabelCandidate] = []

        for det_path in detection_files:
            payload = json.loads(Path(det_path).read_text())
            frame = FrameRecord(**payload["frame"])
            detections = [Detection(**item) for item in payload["detections"]]

            ball_detection = self._select_ball(frame, detections)
            if ball_detection:
                ball_records.append(LabelCandidate(frame, [ball_detection]))
            elif self.include_background_frames:
                ball_records.append(LabelCandidate(frame, []))

        return ball_records

    def _select_ball(self, frame: FrameRecord, detections: List[Detection]) -> Detection | None:
        threshold = float(self.ball_config.get("confidence_threshold", 0.35))
        hsv_check = bool(self.ball_config.get("hsv_check", False))
        hue_range = self.ball_config.get("hsv_hue_range", [300, 360])
        min_saturation = float(self.ball_config.get("min_saturation", 0.5))

        candidates = [det for det in detections if det.class_name == "ball" and det.confidence >= threshold]
        if not candidates:
            return None
        best = max(candidates, key=lambda det: det.confidence)
        if not hsv_check:
            return best

        if self._passes_hsv_check(frame.frame_path, best, hue_range, min_saturation):
            return best
        return None

    def _passes_hsv_check(
        self,
        image_path: str,
        detection: Detection,
        hue_range: List[float],
        min_saturation: float,
    ) -> bool:
        image = cv2.imread(image_path)
        if image is None:
            return False
        h_min, h_max = hue_range
        x1, y1, x2, y2 = map(int, detection.bbox_xyxy)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return False
        crop = image[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0].astype(np.float32) * (360.0 / 180.0)
        sat = hsv[:, :, 1].astype(np.float32) / 255.0
        mean_hue = hue.mean()
        mean_sat = sat.mean()
        return h_min <= mean_hue <= h_max and mean_sat >= min_saturation
