## Part 1 – Dataset Creation Pipeline

This repository contains a reference implementation of the **Part 1 dataset creation pipeline** described in your blueprint. The code now focuses on generating a high-quality, versioned dataset for a lightweight **ball-only YOLO-style detector** destined for real-time tracking.

The implementation is modular so each step can be executed independently or chained into a full pipeline.

### Repository layout

```
configs/
  pipeline_config.yaml         # Default configuration for every stage
dataset_pipeline/
  __init__.py
  catalog.py                   # Metadata models and manifest helpers
  ingestion.py                 # Raw video catalog builder
  frame_extraction.py          # Frame sampling utilities
  teacher_inference.py         # YOLOv8x6 pseudo-labelling logic
  postprocessing.py            # Ball-specific filtering
  label_generation.py          # YOLO-format label writers
  qc.py                        # Quality-control scaffolding
  splitting.py                 # Dataset splitting and versioning
  pipeline.py                  # Orchestrates Part 1 end-to-end
pyproject.toml                 # Python dependencies (ultralytics, opencv, etc.)
```

### Installation

```bash
python -m venv .venv
.venv\Scripts\activate  # or source .venv/bin/activate on macOS/Linux
pip install -e .
```

### Usage overview

1. Drop all MP4 recordings into `data/raw_videos/`. (Optional) You can still organize by `<session>/<camera>/` if desired; the ingestion stage now understands both layouts.
2. Adjust `configs/pipeline_config.yaml` (paths, thresholds, sampling strategy, dataset version).
3. Run individual modules or execute the orchestrator, for example:

```bash
python -m dataset_pipeline.pipeline --config configs/pipeline_config.yaml
```

Each stage persists artifacts (manifests, frames, detections, ball labels, splits) inside the configured workspace so they can be versioned and inspected independently. Review `pipeline.py` for the orchestration order and refer to inline docstrings for detailed behavior.

> **GPU inference:** `configs/pipeline_config.yaml` now defaults to `device: "auto"` for the teacher model. The pipeline prefers CUDA when available, otherwise falls back to Apple Metal (MPS) on supported Macs, then CPU.

## Part 2 – Ball Model Training Pipeline

With `records/ball_dataset_v1.0.json` produced by Part 1 you can now fine-tune a lightweight YOLO detector dedicated to the pink ball.

### Configuration

`configs/training/ball_config.yaml` defines everything the training scripts need:

- Experiment metadata (`name`, `output_dir`, `seed`)
- Dataset manifest path and where to materialize YOLO-friendly symlinks
- Base checkpoint (`yolov8n.pt` by default) and device preference
- Training/augmentation overrides passed straight into Ultralytics’ trainer
- Evaluation preferences (which split to run, extra Ultralytics flags)

### Training

The `train_ball.py` entry point performs three main tasks:

1. Load the dataset manifest and build a YOLO layout under `datasets/ball/training_sets/<version>/<experiment>/images|labels/<split>/...` using symlinks so the raw assets stay in place.
2. Initialize Ultralytics YOLO with the requested lightweight checkpoint and hyperparameters.
3. Launch fine-tuning plus an automatic evaluation pass on the held-out test split once training finishes.

Example workflow:

```bash
# 1) Stage the dataset + launch fine-tuning
python train_ball.py --config configs/training/ball_config.yaml

# 2) (Optional) Inspect the cached label preview Ultralytics saves
open artifacts/training/ball_yolov8n_v1/train/labels.jpg
```

Artifacts land under `artifacts/training/<experiment>/train/` following Ultralytics'
standard structure (checkpoints in `weights/`, TensorBoard in `train/`, plots in `train/`).
Every time you rerun the script the staging directory
`datasets/ball/training_sets/<version>/<experiment>/` is rebuilt from the manifest so
new sessions or split tweaks are automatically reflected. If you only want to restage the
data (e.g., after editing `records/ball_dataset_v1.0.json`) run the training command with
`--epochs 0` so Ultralytics performs the preparation step and exits immediately.

### Standalone evaluation

Use `evaluate_ball.py` to score any checkpoint (produced by `train_ball.py` or otherwise) on the splits defined in the manifest:

```bash
# score the best checkpoint produced by the previous training run
python evaluate_ball.py \
  --config configs/training/ball_config.yaml \
  --weights artifacts/training/ball_yolov8n_v1/train/weights/best.pt
```

Evaluation reports are written to `artifacts/training/ball_yolov8n_v1/test/` (or the
`evaluation.name` override) so you can diff metrics across dataset versions or model
weights. Both scripts reuse the manifest-driven staging step, so they remain in sync as
you iterate on the dataset.

### Stage reference

1. **Raw Data Ingestion** – `dataset_pipeline.ingestion` scans `data/raw_videos/` (either flat MP4s or legacy `<session>/<camera>/` dirs), records video hashes, and writes `records/raw_sessions.yaml`. When using the flat layout, each video inherits `session_id` from its filename and a default `camera_id` of `default_camera`; override either via the `metadata_hook` if needed.
2. **Frame Extraction** – `dataset_pipeline.frame_extraction` samples frames into `images/raw/<session>/<camera>/frame_XXXXXX.jpg` and logs them in `records/frames_index.csv`.
3. **Teacher Inference** – `dataset_pipeline.teacher_inference` runs YOLOv8x6 on every frame, storing detections in JSON plus metadata for reproducibility. The `device` option defaults to `auto`, picking CUDA when available, then Apple MPS, and CPU otherwise, so it works out of the box on machines without discrete GPUs. Use `teacher_inference.alias_map` (default maps `"sports ball"` → `"ball"`) when the checkpoint’s native class names differ from the canonical labels expected downstream.
4. **Ball Filtering** – `dataset_pipeline.postprocessing` applies ball-specific thresholds and optional HSV checks, producing one label candidate per frame.
5. **YOLO Label Export** – `dataset_pipeline.label_generation` normalizes boxes and writes ball-only YOLO label files.
6. **Quality Control** – `dataset_pipeline.qc` emits reviewer queues so manual annotators can prioritize tricky ball frames.
7. **Dataset Splits & Versioning** – `dataset_pipeline.splitting` creates split manifests and `pipeline.py` writes final dataset descriptors for the ball detector (`records/ball_dataset_v1.0.json` by default).

### Visualization helper

After running the pipeline you can inspect individual detections by overlaying them on the source image:

```bash
python visualize_ball_detection.py \
  --detection detections/Trial2-Take5.1-Back/default_camera/frame_000003.json \
  --output artifacts/frame_000003_overlay.jpg
```

The script highlights every detection matching the requested `--label` (default `ball`) with confidence above `--min-confidence`.

### Video tracking utility

Use `track_ball_video.py` to inspect entire MP4s with either the teacher or the fine-tuned detector. The script draws YOLO predictions, optionally wires in a tracker (ByteTrack by default), and writes an annotated video.

```bash
# Trained model: check how the lightweight detector behaves
python track_ball_video.py \
  --input data/raw_videos/TrialSquare-Take2-Back.mp4 \
  --output artifacts/videos/TrialSquare-Take2-Back_trained.mp4 \
  --weights artifacts/training/ball_yolov8n_v1/train/weights/best.pt

# Teacher model: filter only the sports ball class while using ByteTrack
python track_ball_video.py \
  --input data/raw_videos/TrialSquare-Take2-Back.mp4 \
  --output artifacts/videos/TrialSquare-Take2-Back_teacher.mp4 \
  --weights yolov8x6.pt \
  --class-name "sports ball" \
  --tracker bytetrack.yaml
```

Switch `--tracker none` for frame-by-frame detections, or provide `--class-id` when your model
uses a different label ordering. By default videos are saved under `artifacts/videos/`.
