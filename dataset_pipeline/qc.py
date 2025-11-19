from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from .postprocessing import LabelCandidate


class QualityControlPlanner:
    """Creates review queues for annotators based on heuristics."""

    def __init__(
        self,
        output_root: str | Path,
        priority_rules: List[str],
    ) -> None:
        self.output_root = Path(output_root)
        self.priority_rules = priority_rules

    def build(self, ball_records: Iterable[LabelCandidate]) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        ball_queue = self._build_ball_queue(list(ball_records))
        (self.output_root / "ball_review.json").write_text(json.dumps(ball_queue, indent=2))

    def _build_ball_queue(self, records: List[LabelCandidate]) -> List[dict]:
        queue: List[dict] = []
        missing_streak = 0
        for record in records:
            detection = record.detections[0] if record.detections else None
            triggers = []
            if detection:
                missing_streak = 0
                if detection.confidence < 0.45 and "ball_confidence<0.45" in self.priority_rules:
                    triggers.append("ball_confidence<0.45")
            else:
                missing_streak += 1
                if (
                    "ball_missing_for>5_frames" in self.priority_rules
                    and missing_streak >= 5
                ):
                    triggers.append("ball_missing_for>5_frames")
            if triggers:
                queue.append(
                    {
                        "frame": record.frame.to_dict(),
                        "detections": [det.to_dict() for det in record.detections],
                        "triggers": triggers,
                    }
                )
        return queue
