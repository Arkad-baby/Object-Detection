from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("../YOLO_Weights/yolov8l.pt")

# Input video
cap = cv2.VideoCapture("Assets/videos/cars.mp4")

while True:
    success, frame = cap.read()
    if not success:
        break

    # Run YOLO inference
    results = model(frame, stream=True)

    for r in results:
        annotated_frame = r.plot()  # Draw YOLO annotations
    cv2.imshow("Img", annotated_frame)
        # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cap.destroyAllWindows()