from pathlib import Path
from collections import deque
import cv2
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter

CONF_THRES = 0.25
MAX_MISSES = 10
MIN_BOX_AREA = 16  # filter tiny junk
TRAIL_LEN = 200

class DroneTrack:
    def __init__(self, cx: float, cy: float):
        self.kf = self._build_kf(cx, cy)
        self.history = deque(maxlen=TRAIL_LEN)
        self.history.append((int(cx), int(cy)))
        self.missed = 0
        self.last_box = None

    def _build_kf(self, cx, cy):
        # state: [x, y, vx, vy]
        # measurement: [x, y]
        kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0

        kf.x = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)

        kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=np.float32)

        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # Covariances
        kf.P *= 50.0     # initial uncertainty
        kf.R *= 10.0     # measurement noise
        kf.Q = np.array([
            [1, 0, 0.5, 0],
            [0, 1, 0, 0.5],
            [0.5, 0, 2, 0],
            [0, 0.5, 0, 2]
        ], dtype=np.float32)

        return kf

    def predict(self):
        self.kf.predict()
        px, py = self.center
        self.history.append((int(px), int(py)))
        self.missed += 1

    def update(self, cx: float, cy: float):
        z = np.array([[cx], [cy]], dtype=np.float32)
        self.kf.update(z)
        px, py = self.center
        self.history.append((int(px), int(py)))
        self.missed = 0

    @property
    def center(self):
        return float(self.kf.x[0, 0]), float(self.kf.x[1, 0])

def best_detection(results):
    """
    Returns best box as (x1, y1, x2, y2, conf), or None.
    Picks highest-confidence detection.
    """
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return None

    best = None
    best_conf = -1.0

    for b in boxes:
        conf = float(b.conf[0].item())
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
        area = (x2 - x1) * (y2 - y1)
        if area < MIN_BOX_AREA:
            continue
        if conf > best_conf:
            best_conf = conf
            best = (int(x1), int(y1), int(x2), int(y2), conf)

    return best

def draw_trail(frame, history):
    pts = list(history)
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i - 1], pts[i], (0, 255, 255), 2)

def process_video(video_path: Path, model: YOLO, detections_dir: Path, output_dir: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = output_dir / f"{video_path.stem}_tracked.mp4"
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    video_det_dir = detections_dir / video_path.stem
    video_det_dir.mkdir(parents=True, exist_ok=True)

    track = None
    frame_idx = 0
    written_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            conf=CONF_THRES,
            verbose=False
        )[0]

        det = best_detection(results)

        drone_present = det is not None

        if track is None and det is not None:
            x1, y1, x2, y2, conf = det
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            track = DroneTrack(cx, cy)
            track.last_box = (x1, y1, x2, y2)

        elif track is not None:
            track.predict()

            if det is not None:
                x1, y1, x2, y2, conf = det
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                track.update(cx, cy)
                track.last_box = (x1, y1, x2, y2)
            elif track.missed > MAX_MISSES:
                track = None

        # Save only frames where drone is present by detector
        if drone_present:
            save_path = video_det_dir / f"frame_{frame_idx:05d}.jpg"
            cv2.imwrite(str(save_path), frame)

        # Output only frames where drone is present OR recently tracked
        keep_frame = drone_present or (track is not None and track.missed <= MAX_MISSES)

        if keep_frame:
            vis = frame.copy()

            if det is not None:
                x1, y1, x2, y2, conf = det
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    vis,
                    f"drone {conf:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            if track is not None:
                px, py = map(int, track.center)
                cv2.circle(vis, (px, py), 4, (0, 0, 255), -1)
                draw_trail(vis, track.history)

            writer.write(vis)
            written_frames += 1

        frame_idx += 1

    cap.release()
    writer.release()
    print(f"{video_path.name}: wrote {written_frames} frames -> {out_path}")

def main():
    video_dir = Path("Videos")
    detections_dir = Path("detections")
    output_dir = Path("tracks")

    detections_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO("runs/detect/train6/weights/best.pt")

    for video_path in sorted(video_dir.glob("*.mp4")):
        process_video(video_path, model, detections_dir, output_dir)

if __name__ == "__main__":
    main()