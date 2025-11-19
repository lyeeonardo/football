from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import cv2
import pandas as pd

from .catalog import FrameRecord, VideoRecord


class FrameExtractor:
    """Extracts frames following the sampling strategy defined in the config."""

    def __init__(
        self,
        frames_root: str | Path,
        sampling_strategy: str = "stride",
        stride: int = 3,
        target_fps: int = 10,
        image_format: str = "jpg",
        quality: int = 95,
        index_path: str | Path | None = None,
    ) -> None:
        self.frames_root = Path(frames_root)
        self.sampling_strategy = sampling_strategy
        self.stride = max(stride, 1)
        self.target_fps = target_fps
        self.image_format = image_format
        self.quality = quality
        self.index_path = Path(index_path) if index_path else None

    def extract(self, videos: Iterable[VideoRecord]) -> List[FrameRecord]:
        records: List[FrameRecord] = []
        for video in videos:
            records.extend(self._process_video(video))
        if self.index_path:
            self._write_index(records)
        return records

    def _process_video(self, video: VideoRecord) -> List[FrameRecord]:
        capture = cv2.VideoCapture(str(video.video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video {video.video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frame_records: List[FrameRecord] = []
        next_keep = 0
        keep_every = self._compute_stride(fps)

        frame_index = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_index < next_keep:
                frame_index += 1
                continue
            frame_path = self._frame_path(video, frame_index)
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            params = []
            if self.image_format.lower() in {"jpg", "jpeg"}:
                params = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
            cv2.imwrite(str(frame_path), frame, params)
            timestamp = frame_index / fps
            frame_records.append(
                FrameRecord(
                    session_id=video.session_id,
                    camera_id=video.camera_id,
                    video_path=video.video_path,
                    frame_path=str(frame_path),
                    frame_index=frame_index,
                    timestamp_sec=timestamp,
                )
            )
            frame_index += 1
            next_keep += keep_every
        capture.release()
        return frame_records

    def _compute_stride(self, fps: float) -> int:
        if self.sampling_strategy == "fps":
            if self.target_fps <= 0:
                return 1
            return max(int(round(fps / self.target_fps)), 1)
        return self.stride

    def _frame_path(self, video: VideoRecord, frame_index: int) -> Path:
        filename = f"frame_{frame_index:06d}.{self.image_format}"
        return self.frames_root / video.session_id / video.camera_id / filename

    def _write_index(self, records: List[FrameRecord]) -> None:
        rows = [record.to_dict() for record in records]
        df = pd.DataFrame(rows)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.index_path, index=False)
