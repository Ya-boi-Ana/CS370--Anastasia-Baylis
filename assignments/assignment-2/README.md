# Assignment 2 – Image-to-Video Semantic Retrieval via Object Detection
### By Anastasia Baylis
### CS370-102

## Hugging Face Link (Includes README.md)
https://huggingface.co/datasets/Ya-boi-ana/cs370-assignment-2-detections

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

## Report
A link to the report: https://docs.google.com/document/d/1L2sc_TiwtAyxnmVGcYm2kwovzRTaxrdoG4yRoORAL50/edit?usp=sharing

as well as the full report here:

#                    Assignment 2 Image-to-Video Semantic Retrieval via Object Detection

##                          Prepared for: Professor Pantelis Monogioudis
##                          Prepared by: Anastasia Baylis
##                          CS370-102
##                          2/19/2026

The goal of this assignment is to build an image-to-video semantic retrieval system using object detection and structured indexing. Given a query image containing any exterior car component and a video of a car, the system should properly identify each given component visible in the given image, and retrieve video segments where these car parts appear. 

I implemented a pipeline that samples frames from the video, 1 frame per second, detects the car’s components using a YOLOv8 model that was trained on the given Ultralytics car-part dataset, retrieves frames from the video based on the query images, and stores all detections in a parquet index.

I chose a YOLOv8 model and trained it on the given Ultralytics car-part dataset. I chose the YOLOv8 model because it was easy to deploy and was recommended for frame-by-frame processing. The given dataset contained labeled images of exterior car parts, and I chose to do segmentation because it is more accurate than bounding box only detection. To train the model I chose to do 30 epochs with an input image resolution of 640 as well as the default Ultralytics parameters, and after training I saved the best “checkpoint” (best.pt) to be used.

My video sampling strategy was to sample the input video at 1 frame per second using ffmpeg, a free software to process video or image files, which produced about 2,800 frames for the entire video. Then each frame was processed by my trained YOLOv8 segmentation model,  each time a component was detected it would record then store into a parquet file: video_id, timestamp_sec, class_label, confidence_score, bounding box coordinates, and the detector_name. Each row of the parquet file represents a detection which ended up being around 6,900 detections with 2,160 unique timestamps.

The matching logic I used was to load query images from the hugging face dataset. Each of these query images passed through the trained YOLOv8 segmentation model that identified the visible car parts. These identified parts were matched using the class labels from the queried images that were stored in the video detection index. All video frames with the same class label had their time stamps retrieved by: start timestamp, end timestamp, class label, and the number of supporting detections. For example, if a query were to detect “front_bumper” it would detect multiple intervals in the video where the front bumper is visible, where these intervals can be any length (from 1 second to the entire video).

The trained model successfully detected multiple exterior components including: the front bumper, front glass, hood, mirrors, and doors. The query images often had coherent retrieval intervals that would last from several seconds to minutes, depending on if the component was visible. Larger components had better and longer intervals, while smaller components had shorter intervals. Overall the system was able to correctly detect the queried parts and return correct time ranges without the use of manual intervention.

There are multiple failure cases that have happened while conducting this assignment. One failure is query images coming from a different distribution, and not the specified video, which can cause missed detections or incorrect detections. Retrieval relies only on class labels overlapping, not the similarity scores. Because I sampled at 1 frame per second, this could cause the detector to miss short appearances of car parts, like if a mirror were only in frame for 1 frame, not 2-5. Another issue is YOLOv8 itself, it may not be able to detect parts if they are partially covered, say if the man in the video was in front of the drivers side door or in front of part of the front bumper, although it is partially visible it is not fully visible to be detected.

Overall, this assignment demonstrated a semantic image-to-video retrieval pipeline based on object detection and structured indexing. By combining a segmentation model with a parquet based detection index, the system was able to efficiently retrieve relevant video segments from the still-image queries. Some future improvements may be: higher frame sampling rates, higher quality query images, and the use of multiple labels to determine the accuracy of a part.
