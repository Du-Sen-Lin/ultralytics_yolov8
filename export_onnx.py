import torch
from ultralytics import YOLO

# 加载训练好的模型
model_path = '/root/project/research/Yolo/ultralytics/runs/detect/train7/weights/best.pt'
model = YOLO(model_path)

# 设置导出路径
# export_path = 'F:/Wood/Pycharm_workspace/ultralytics/runs/detect/train7/weights/equal_diameter_yolov8s_dynamic_20240524.onnx'

# 导出为 ONNX 格式
# TensorRT
# model.export(format='onnx', dynamic=True, simplify=False, opset=11)
model.export(format='onnx', opset=12)

# openvino
# model.export(format='onnx', opset=11)
print(f"Model has been converted and saved")
