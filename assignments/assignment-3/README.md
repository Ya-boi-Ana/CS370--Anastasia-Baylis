# Assignment 3
### By: Anastasia Baylis
### CS370-102

## Hugging Face:
https://huggingface.co/datasets/Ya-boi-ana/CS370-Assignment-3

## Youtube videos:
Video 1: https://youtu.be/nt8S4SusHBU
Video 2: https://youtu.be/LCS266J2BaM

## Dataset Choice + Detector Config

I chose a drone object detector dataset from Roboflow Universe(https://universe.roboflow.com/uav-qnoms/uav-wqshy/dataset/6), which has labeled UAV bounding boxes. I exported it to YOLOv8 formatting to train directly with the ultralytics YOLOv8n framework. The Model was initialized with pre-trained weights, and was fine-tuned for a single drone class using 30 epochs and an image size of 640 and in batches of 8 to improve the detection of drones. The trained model will output bounding boxes with confidence scores when a drone is detected in each frame, the center of the bounding box is used as a measurement input for the tracking system.

## Kalman filter design + noise

A Kalman filter was implemented using the Filterpy library to track the drone across frames. the filter uses the vector: [x,y,vx,vy], where x and y represent the pixel coordinates of the drones bounding box center, and vx,vy represent the estimated velocity of the drone. Giving us the measurement vector z=[x,y]. A constant velocity motion model has been implemented to predict the next possible position of the drone in the next frame using the current position and velocity: xt+1 = xt+vx, yt+1 = yt+vy.

The noise parameters: 
state covariance (P) is initialized with large values to measure the uncertainty in the initial drone position and velocity. 
Measurement noise (R) represents the uncertainty in the detector output, this is so when the bounding box predictions fluctuate that a small fluctuation doesn't cause the box to over-compensate for that movement.
Process Noise(Q) models the different possibilities of drone motion between frames small process noise values allow the tracker to adapt to the motion changes, and smooth the trajectory.


## Failure cases:

Some failure cases are: 
* the drone being too small in frame (think only a few pixels)
* motion blur when the drone is moving quickly
* Other flying background objects like birds or planes
* lighting changes

When the detector misses the drone in certain frames, the Kalman filter will keep predicting the drones position and movement using the motion model. Say if the drone goes fast in a straight line, and the detector misses it, the box will appear again not long after because the kalman filter predicted it going straight still. If the detector isn't able to find the drone again after a certain amount of time the tracker will terminate. This allows the system to maintain good tracking even if the detector's output is noisy or missing.
