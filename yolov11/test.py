from ultralytics import YOLO

# model = YOLO("yolo11s.pt")
model_path = "/root/project/research/Yolo/ultralytics_yolov11/ultralytics/runs/detect/train2/weights/best.pt"
model = YOLO(model_path)

# 1、Customize validation settings
# data_path = "/root/project/research/Yolo/ultralytics_yolov11/ultralytics/ultralytics/cfg/datasets/glass.yaml"
# validation_results = model.val(data=data_path, imgsz=640, batch=16, conf=0.25, iou=0.6, device="0")

# 2、Validate the model
# metrics = model.val()
# print(metrics.box.map)  # map50-95

# 3、Process results list
results = model(["/root/dataset/glass_data_0930/images/val/Image_20240725165715498.bmp", "/root/dataset/glass_data_0930/images/val/Image_20240725170244714.bmp"])
for result in results:
   boxes =  result.boxes
   masks = result.masks
   keypoints = result.keypoints
   probs = result.probs
   obb = result.obb
   result.show()
   result.save(filename = "result.jpg")
   # print(f"boxes: {boxes}")
   print(f"masks: {masks}")
