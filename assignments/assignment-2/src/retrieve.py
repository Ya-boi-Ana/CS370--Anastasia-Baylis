from __future__ import annotations

import pandas as pd
from datasets import load_dataset
from ultralytics import YOLO

from utils import merge_timestamps

PARQUET_PATH = "outputs/video_detections.parquet"

# Match your video sampling (fps=1)
MAX_GAP_SEC = 2

CONF_QUERY = 0.15
CONF_INDEX = 0.25

# How many labels from the query to use
TOP_K_LABELS = 2


def main() -> None:
    df = pd.read_parquet(PARQUET_PATH)
    if df.empty:
        print("Parquet is empty. Run indexing first.")
        return

    ds = load_dataset("aegean-ai/rav4-exterior-images", split="train")
    q = ds[0]
    img = q["image"]

    print("Query timestamp (from dataset):", q.get("timestamp"), "sec:", q.get("timestamp_sec"))

    model = YOLO("runs/segment/train2/weights/best.pt")

    # Detect labels in query image
    res = model.predict(source=img, conf=CONF_QUERY, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        print("No detections in query image.")
        return

    dets = []
    for b in res.boxes:
        cls_id = int(b.cls.item())
        conf = float(b.conf.item())
        label = model.names.get(cls_id, str(cls_id))
        dets.append((label, conf))

    # sort by confidence and take top-k unique labels
    dets.sort(key=lambda x: x[1], reverse=True)
    labels = []
    for label, conf in dets:
        if label not in labels:
            labels.append(label)
        if len(labels) >= TOP_K_LABELS:
            break

    print("Query detected labels:", labels)

    # Retrieve intervals for each label
    for label in labels:
        hits = df[(df["class_label"] == label) & (df["confidence_score"] >= CONF_INDEX)]
        ts = hits["timestamp_sec"].tolist()

        intervals = merge_timestamps(ts, max_gap_sec=MAX_GAP_SEC)

        print(f"\nLabel: {label} | supporting detections: {len(hits)}")
        if not intervals:
            print("  No matching intervals found.")
            continue

        for itv in intervals[:10]:
            print({
                "start_timestamp": itv.start,
                "end_timestamp": itv.end,
                "class_label": label,
                "number_of_supporting_detections": itv.support
            })


if __name__ == "__main__":
    main()
