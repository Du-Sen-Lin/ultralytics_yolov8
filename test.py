from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('/root/project/research/Yolo/ultralytics/runs/detect/train7/weights/best.pt ')

# Define path to the image file
source = '/root/dataset/glass_data/images/val/Image_20240725165715498.bmp'

# Run inference on the source
results = model(source)  # list of Results objects
# results = model.predict(source, save=True, imgsz=640, conf=0.5)
# print(f"results: {results}")
for r in results:
    print(f"box: {r.boxes}")