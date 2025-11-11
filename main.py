from ultralytics import YOLO
import cv2
model=YOLO(model="../YOLO_Weights/yolov8l.pt") #version 8 nano

result=model(source="Assets/videos/cars.mp4",show=True,save=True)
cv2.waitKey(0)