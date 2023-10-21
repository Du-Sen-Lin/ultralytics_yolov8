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
