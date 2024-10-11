from ultralytics import YOLO

model = YOLO('Project/Models/model1/weights/best.pt')
model.val()