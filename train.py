from ultralytics import YOLO
from importlib_metadata import metadata

# Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO('/root/project/bp_algo/common/YOLO/ultralytics_yolov8/ultralytics/cfg/models/v8/hayao_yolov8s.yaml').load(
    '/root/project/bp_algo/common/YOLO/ultralytics_yolov8/yolov8s.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(
    data='/root/project/bp_algo/common/YOLO/ultralytics_yolov8/ultralytics/cfg/datasets/hayao.yaml ', epochs=50, imgsz=640)
