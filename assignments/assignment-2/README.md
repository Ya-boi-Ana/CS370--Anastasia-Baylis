# Assignment 2 – Image-to-Video Semantic Retrieval via Object Detection
### By Anastasia Baylis
### CS370-102

## Quick description
Detections were generated using a YOLOv8 segmentation model fine-tuned on the Ultralytics Car Parts dataset.
Video sampled at 1 FPS.

This dataset repository contains detection outputs (one row per detection) for a car exterior video, used for image-to-video semantic retrieval.

~2800 frames indexed

## Folder layout
- `data/`: source video download
- `frames/`: extracted frames from the video corpus
- `src/`: indexing + retrieval scripts
- `outputs/`: Parquet detection index + retrieval outputs
- `report/`: writeup

## Retrieval output (per query image)
For each query, we return:
- `start_timestamp`
- `end_timestamp`
- `class_label used for retrieval`
- `number_of_supporting_detections`

## Schema
- `video_id` (string): YouTube video id
- `timestamp_sec` (int): timestamp in seconds
- `frame_file` (string): frame filename
- `class_label` (string): car-part label (e.g., front_bumper, hood)
- `confidence_score` (float)
- `x_min, y_min, x_max, y_max` (float): bounding box coordinates
- `detector_name` (string): model identifier
