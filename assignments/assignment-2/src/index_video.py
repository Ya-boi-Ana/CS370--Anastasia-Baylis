from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

VIDEO_ID = "YcvECxtXoxQ"

FRAMES_DIR = Path("frames")
OUT_PARQUET = Path("outputs/video_detections.parquet")

# Frame sampling rate used in ffmpeg. You extracted fps=1.
FPS = 1

# Detection settings
CONF_THRES = 0.25


MAX_FRAMES = None


def frame_index_from_name(name: str) -> int:
    m = re.search(r"frame_(\d+)\.(jpg|jpeg|png)$", name)
    if not m:
        raise ValueError(f"Unexpected frame filename: {name}")
    return int(m.group(1)) - 1  # frame_000001 -> index 0


def main() -> None:
    frame_paths = sorted(FRAMES_DIR.glob("frame_*.jpg"))
    if not frame_paths:
        raise SystemExit(f"No frames found in {FRAMES_DIR}. Extract frames first.")

    if MAX_FRAMES is not None:
        frame_paths = frame_paths[:MAX_FRAMES]

    # TODO: swap this to a car-parts model later (part-level classes)
    model = YOLO("runs/segment/train2/weights/best.pt")

    rows = []
    for fp in tqdm(frame_paths, desc="Detecting frames"):
        frame_idx = frame_index_from_name(fp.name)
        timestamp_sec = int(round(frame_idx / FPS))

        res = model.predict(source=str(fp), conf=CONF_THRES, verbose=False)[0]
        if res.boxes is None:
            continue

        for b in res.boxes:
            cls_id = int(b.cls.item())
            conf = float(b.conf.item())
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            class_label = model.names.get(cls_id, str(cls_id))

            rows.append({
                "video_id": VIDEO_ID,
                "timestamp_sec": timestamp_sec,
                "frame_file": fp.name,
                "class_label": class_label,
                "confidence_score": conf,
                "x_min": x1,
                "y_min": y1,
                "x_max": x2,
                "y_max": y2,
                "detector_name": "yolov8n-seg_carparts_train2"
            })

    df = pd.DataFrame(rows, columns=[
        "video_id", "timestamp_sec", "frame_file",
        "class_label", "confidence_score",
        "x_min", "y_min", "x_max", "y_max",
        "detector_name"
    ])

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)

    print(f"\nWrote {len(df)} detections -> {OUT_PARQUET}")
    if len(df) > 0:
        print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
