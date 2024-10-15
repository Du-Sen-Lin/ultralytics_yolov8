from ultralytics import YOLO
from importlib_metadata import metadata

# Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO('/root/project/research/Yolo/ultralytics/ultralytics/cfg/models/v8/glass_yolov8s.yaml').load(
    '/root/project/research/Yolo/ultralytics/yolov8s.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(
    data='/root/project/research/Yolo/ultralytics/ultralytics/cfg/datasets/glass.yaml', epochs=300, imgsz=640)
