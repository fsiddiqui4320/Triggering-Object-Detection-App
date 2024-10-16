from ultralytics import YOLO
import cv2  

# This code only checks agaisnt one image, and displays the bouding boxs detected
# This does not test agaisnt the actual test images (will add later)

model = YOLO('Project/Models/model1/weights/best.pt')

image_path = 'Project/Data/test/images/0b891eae-ep261_jpg.rf.6f5015ea3da9a8ba995acda22945eb98.jpg'

results = model(image_path)

imgResult = results[0].plot()
img_rgb = cv2.cvtColor(imgResult, cv2.COLOR_BGR2RGB)
cv2.imshow('Detection Results', img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()