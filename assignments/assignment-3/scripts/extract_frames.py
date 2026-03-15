from pathlib import Path
import cv2

def extract_frames_from_video(video_path: Path, out_dir: Path, fps_sample: float = 5.0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = 30.0

    step = max(int(round(src_fps / fps_sample)), 1)
    video_stem = video_path.stem
    video_out = out_dir / video_stem
    video_out.mkdir(parents=True, exist_ok=True)

    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frame_path = video_out / f"frame_{saved:05d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            saved += 1
        idx += 1

    cap.release()
    print(f"{video_path.name}: saved {saved} frames to {video_out}")

def main():
    video_dir = Path("data/raw_videos")
    out_dir = Path("data/frames")
    out_dir.mkdir(parents=True, exist_ok=True)

    for video_path in video_dir.glob("*.mp4"):
        extract_frames_from_video(video_path, out_dir, fps_sample=5.0)

if __name__ == "__main__":
    main()