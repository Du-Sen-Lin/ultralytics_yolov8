# Yolov8

fork by 2023.10.16

## 一、实验测试

```
Doc: https://docs.ultralytics.com/modes/train/
```

```python
# 下载代码
git clone https://github.com/ultralytics/ultralytics
cd ultralytics
pip install -e .
```

```python
# 测试
yolo predict model=yolov8s.pt source=../yolov5/data/images/bus.jpg

# 训练 lvqi_s1.yaml lvqi_s1_yolov8.yaml
# Build a new model from YAML and start training from scratch
yolo detect train data=coco128.yaml model=yolov8n.yaml epochs=100 imgsz=640
# Start training from a pretrained *.pt model
yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640
# Build a new model from YAML, transfer pretrained weights to it and start training
yolo detect train data=coco128.yaml model=yolov8n.yaml pretrained=yolov8n.pt epochs=100 imgsz=640

yolo detect train data=/root/project/bp_algo/common/YOLO/ultralytics_yolov8/ultralytics/cfg/datasets/hayao.yaml model=/root/project/bp_algo/common/YOLO/ultralytics_yolov8/ultralytics/cfg/models/v8/hayao_yolov8s.yaml pretrained=/root/project/bp_algo/common/YOLO/ultralytics_yolov8/yolov8s.pt epochs=50 imgsz=640


# 部署参考
https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/vision/detection/yolov8
```

##### **YOLOv8在COCO上的精度**：

See [Detection Docs](https://docs.ultralytics.com/tasks/detect/) for usage examples with these models trained on [COCO](https://docs.ultralytics.com/datasets/detect/coco/), which include 80 pre-trained classes.

| Model                                                        | size (pixels) | mAPval 50-95 | Speed CPU ONNX (ms) | Speed A100 TensorRT (ms) | params (M) | FLOPs (B) |
| ------------------------------------------------------------ | ------------- | ------------ | ------------------- | ------------------------ | ---------- | --------- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) | 640           | 37.3         | 80.4                | 0.99                     | 3.2        | 8.7       |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt) | 640           | 44.9         | 128.4               | 1.20                     | 11.2       | 28.6      |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) | 640           | 50.2         | 234.7               | 1.83                     | 25.9       | 78.9      |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt) | 640           | 52.9         | 375.2               | 2.39                     | 43.7       | 165.2     |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) | 640           | 53.9         | 479.1               | 3.53                     | 68.2       | 257.8     |

- **mAPval** values are for single-model single-scale on [COCO val2017](http://cocodataset.org/) dataset.
  Reproduce by `yolo val detect data=coco.yaml device=0`
- **Speed** averaged over COCO val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/) instance.
  Reproduce by `yolo val detect data=coco.yaml batch=1 device=0|cpu`

##### train.py

```python
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

```

##### test.py

```shell
yolo predict model=/root/project/bp_algo/common/YOLO/ultralytics_yolov8/runs/detect/train10/weights/best.pt source=/root/dataset/public/object_detect/dataset_yolo_hayao/dataset/images/val/Image_20230310171144605.bmp
```



```python
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('/root/project/bp_algo/common/YOLO/ultralytics_yolov8/runs/detect/train10/weights/best.pt ')

# Define path to the image file
source = '/root/dataset/public/object_detect/dataset_yolo_hayao/dataset/images/val/Image_20230310171144605.bmp'

# Run inference on the source
# results = model(source)  # list of Results objects
results = model.predict(source, save=True, imgsz=640, conf=0.5)
print(f"results: {results}")
for r in results:
    print(f"box: {r.boxes}")
```



## 二、源码模块分析





# What's New

##### Oct 15, 2024: 

增加 ./yolov11 
