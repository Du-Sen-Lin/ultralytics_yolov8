from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n.yaml")  # build a new model from YAML
# model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
# model = YOLO("/root/project/research/Yolo/ultralytics_yolov11/ultralytics/ultralytics/cfg/models/11/glass_yolov11n.yaml").load(
#     "/root/project/research/Yolo/ultralytics_yolov11/ultralytics/yolo11n.pt")  # build from YAML and transfer weights

model = YOLO("/root/project/research/Yolo/ultralytics_yolov11/ultralytics/ultralytics/cfg/models/11/glass_yolov11s.yaml").load(
   "/root/project/research/Yolo/ultralytics_yolov11/ultralytics/yolo11s.pt")  # build from YAML and transfer weights

# Train the model
data_path = "/root/project/research/Yolo/ultralytics_yolov11/ultralytics/ultralytics/cfg/datasets/glass.yaml"
results = model.train(data=data_path, epochs=100, imgsz=640)