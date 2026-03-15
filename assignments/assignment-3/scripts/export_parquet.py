from pathlib import Path
import pandas as pd

detections_root = Path("detections")
rows = []

for img_path in detections_root.rglob("*.jpg"):
    with open(img_path, "rb") as f:
        rows.append({
            "video": img_path.parent.name,
            "file_name": img_path.name,
            "image_bytes": f.read(),
            "label": "drone_present"
        })

df = pd.DataFrame(rows)
out_path = "detections_samples.parquet"
df.to_parquet(out_path, index=False)

print(f"Saved {out_path} with {len(df)} rows")