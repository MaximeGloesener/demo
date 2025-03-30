from ultralytics import YOLO
import cv2

model = YOLO('best.pt')

image = 'r2.jpg'

results = model(image, imgsz=640, conf=0.4)

out = results[0].plot()

output_path = "output.jpg"
cv2.imwrite(output_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

results[0].show()
