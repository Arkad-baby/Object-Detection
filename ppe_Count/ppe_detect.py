from ultralytics import YOLO
import cv2
import cvzone

model = YOLO("YOLO_Weights/ppe.pt")
cap = cv2.VideoCapture("Assets/videos/ppe-3-1.mp4")

classes = [
    "Hardhat",
    "Mask",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Person",
    "Safety Cone",
    "Safety Vest",
    "Machinery",
    "Vehicle",
]


while True:
    success, frame = cap.read()

    if not success:
        break

    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            h = y2 - y1
            w = x2 - x1
            cls_id = int(box.cls[0])
            conf = box.conf[0]
            className = classes[cls_id]

            cvzone.putTextRect(
                frame,
                f"{className} {conf:.2f}",
                (max(0, int(x1)), max(35, int(y1))),
                scale=0.8,
                thickness=1,
                colorB=(0, 0, 255),
                offset=2,
            )

            cvzone.cornerRect(frame, (int(x1), int(y1), int(w), int(h)), rt=3, l=9, t=3)

    cv2.imshow("Image", frame)

    if cv2.waitKey(1) and 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()