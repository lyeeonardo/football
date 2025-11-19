from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

from rich.console import Console

from training.utils import resolve_device

try:  # pragma: no cover - heavy dependency
    import cv2
except Exception as exc:  # pragma: no cover
    cv2 = None  # type: ignore
    _cv2_error = exc
else:
    _cv2_error = None

try:  # pragma: no cover - heavy dependency
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
    YOLO = None  # type: ignore
    _yolo_error = exc
else:
    _yolo_error = None


console = Console()
DEFAULT_WEIGHTS = Path("artifacts/training/ball_yolov8n_v1/train/weights/best.pt")
DEFAULT_TRACKER = "bytetrack.yaml"


def main() -> None:
    args = parse_args()
    ensure_dependencies()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device, console)
    console.log(f"[bold]Loading model[/bold] {args.weights} on {device}")
    model = YOLO(str(args.weights))
    target_class_id, label_text = resolve_target_class(
        model, args.class_name, args.class_id, args.label
    )
    console.log(f"Tracking class id {target_class_id} ({label_text})")

    fps, width, height = read_video_metadata(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    tracker_config = normalize_tracker_arg(args.tracker)
    console.log(f"[bold]Processing[/bold] {input_path} → {output_path}")
    try:
        if tracker_config:
            total_frames, detected_frames = process_with_tracker(
                model=model,
                input_path=input_path,
                writer=writer,
                tracker_config=tracker_config,
                device=device,
                conf_threshold=args.conf,
                class_id=target_class_id,
                label=label_text,
            )
        else:
            total_frames, detected_frames = process_with_detector(
                model=model,
                input_path=input_path,
                writer=writer,
                device=device,
                conf_threshold=args.conf,
                class_id=target_class_id,
                label=label_text,
            )
    finally:
        writer.release()

    console.log(
        f"[green]Done[/green] – wrote {total_frames} frames "
        f"({detected_frames} with detections) to {output_path}"
    )


def annotate_frame(
    model: "YOLO",
    frame,
    *,
    device: str,
    conf_threshold: float,
    class_id: int,
    label: str,
) -> Tuple:
    result = model.predict(
        frame,
        conf=conf_threshold,
        device=device,
        verbose=False,
    )[0]
    boxes = result.boxes
    best_xyxy = None
    best_conf = -1.0
    if boxes is not None:
        for xyxy, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
            if int(cls) != class_id:
                continue
            if float(conf) > best_conf:
                best_conf = float(conf)
                best_xyxy = xyxy
    if best_xyxy is None:
        overlay_no_detection(frame, label)
        return frame, False
    x1, y1, x2, y2 = [int(float(v)) for v in best_xyxy]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
    cv2.putText(
        frame,
        f"{label} {best_conf:.2f}",
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 165, 255),
        2,
        cv2.LINE_AA,
    )
    return frame, True


def overlay_no_detection(frame, label: str) -> None:
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


def ensure_dependencies() -> None:
    if cv2 is None:
        raise ImportError(
            "opencv-python is required for video IO. Install it via `pip install opencv-python`."
        ) from _cv2_error
    if YOLO is None:
        raise ImportError(
            "ultralytics is required for inference. Install it via `pip install ultralytics`."
        ) from _yolo_error


def process_with_detector(
    model: "YOLO",
    input_path: Path,
    writer: "cv2.VideoWriter",
    *,
    device: str,
    conf_threshold: float,
    class_id: int,
    label: str,
) -> Tuple[int, int]:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    total_frames = 0
    detected_frames = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            total_frames += 1
            annotated, detected = annotate_frame(
                model=model,
                frame=frame,
                device=device,
                conf_threshold=conf_threshold,
                class_id=class_id,
                label=label,
            )
            if detected:
                detected_frames += 1
            writer.write(annotated)
    finally:
        cap.release()
    return total_frames, detected_frames


def process_with_tracker(
    model: "YOLO",
    input_path: Path,
    writer: "cv2.VideoWriter",
    *,
    tracker_config: str,
    device: str,
    conf_threshold: float,
    class_id: int,
    label: str,
) -> Tuple[int, int]:
    stream = model.track(
        source=str(input_path),
        tracker=tracker_config,
        stream=True,
        conf=conf_threshold,
        device=device,
        persist=True,
        verbose=False,
    )
    total_frames = 0
    detected_frames = 0
    for result in stream:
        annotated, detected = annotate_tracking_result(result, class_id=class_id, label=label)
        writer.write(annotated)
        total_frames += 1
        if detected:
            detected_frames += 1
    return total_frames, detected_frames


def annotate_tracking_result(result, *, class_id: int, label: str) -> Tuple:
    frame = result.orig_img.copy()
    detected = False
    boxes = getattr(result, "boxes", None)
    if boxes is not None:
        ids = getattr(boxes, "id", None)
        for idx, (xyxy, conf, cls) in enumerate(zip(boxes.xyxy, boxes.conf, boxes.cls)):
            if int(cls) != class_id:
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
        overlay_no_detection(frame, label)
    return frame, detected


def read_video_metadata(path: Path) -> Tuple[float, int, int]:
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


def normalize_tracker_arg(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"", "none", "off", "false"}:
        return None
    return value


def resolve_target_class(
    model: "YOLO",
    class_name: str | None,
    class_id: int | None,
    label_override: str | None,
) -> Tuple[int, str]:
    names_attr = getattr(model, "names", {})
    if isinstance(names_attr, dict):
        names_map = {int(k): v for k, v in names_attr.items()}
    elif isinstance(names_attr, list):
        names_map = {idx: name for idx, name in enumerate(names_attr)}
    else:
        names_map = {}

    if class_id is not None:
        label_text = label_override or names_map.get(class_id, f"class_{class_id}")
        return class_id, label_text

    if not class_name:
        raise ValueError("Either --class-name or --class-id must be provided.")
    lookup = class_name.lower()
    for idx, name in names_map.items():
        if name.lower() == lookup:
            label_text = label_override or name
            return idx, label_text
    raise ValueError(
        f"Class '{class_name}' not found in model names: {list(names_map.values()) or 'unknown'}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the trained ball model on a video.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw_videos/TrialSquare-Take2-Back.mp4",
        help="Path to the source MP4.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/videos/TrialSquare-Take2-Back_tracked.mp4",
        help="Where to save the annotated video.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help="Path to the trained YOLO weights.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Device to run inference on ("cuda", "mps", "cpu", or "auto").',
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=None,
        help="Class index to track (overrides --class-name when provided).",
    )
    parser.add_argument(
        "--class-name",
        type=str,
        default="ball",
        help="Class name to track (looked up in the model metadata).",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional override for the text drawn on bounding boxes.",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default=DEFAULT_TRACKER,
        help="Tracker config to use (e.g., 'bytetrack.yaml'). Set to 'none' to disable tracking.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
