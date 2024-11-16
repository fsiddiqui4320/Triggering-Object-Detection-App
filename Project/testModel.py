from ultralytics import YOLO
import cv2  

# This code only checks agaisnt one image, and displays the bouding boxs detected
# This does not test agaisnt the actual test images (will add later)

model = YOLO('Project/Models/model1/weights/best.pt')
image_path = 'Project/static/uploads/SIG_Pro_by_Augustas_Didzgalvis.jpg'
image = cv2.imread(cv2.samples.findFile(image_path))
cv2.imshow("New Image", image)
results = model(image_path)
#print(results.boxes)
if isinstance(results, list):
    results = results[0]

for box in results.boxes:
    # Convert tensor to integers
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    
    # Extract the region of interest (ROI) from the image
    roi = image[y1:y2, x1:x2]
    
    # Apply a blur to the ROI
    blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
    
    # Replace the original ROI in the image with the blurred ROI
    image[y1:y2, x1:x2] = blurred_roi

#imgResult = results[0].plot()
#img_rgb = cv2.cvtColor(imgResult, cv2.COLOR_BGR2RGB)
#cv2.imshow('Detection Results', img_rgb)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
cv2.imshow("Blurred Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()