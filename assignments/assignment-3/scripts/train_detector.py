from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="data.yaml",
    epochs=30,
    imgsz=640,
    batch=8,
    workers=0,
    cache=False,
    project="runs",
    name="drone_train_better"
)