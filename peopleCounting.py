from ultralytics import YOLO
import cv2
import cvzone
from Assets.sort import *

model = YOLO("YOLO_Weights/yolov8l.pt")
cap = cv2.VideoCapture("Assets/videos/people2.mp4")

# tracking
# SORT -> Simple Online Realtime Tracking
# keeps it alive for 60 frames even if temporarily lost -> max_age=60
# minimum number of detections before a track is confirmed -> min_hits
# iou_threshold->  Lower values (like 0.3) = more forgiving → more matches.
tracker = Sort(max_age=60, min_hits=3, iou_threshold=0.3)
classes = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

Counter = {"id": [], "totalCount": 0}

while True:
    success, frame = cap.read()

    if not success:
        break

    results = model(frame, stream=True)
    detections = np.empty((0, 5))
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

            if className == "person":
                # cvzone.putTextRect(
                #     frame,
                #     f"{className} {conf:.2f}",
                #     (max(0, x1), max(0, y1)),
                #     scale=0.8,
                #     thickness=1,
                #     colorB=(0, 0, 255),
                #     offset=2,
                # )
                cvzone.cornerRect(frame, (int(x1), int(y1), int(w), int(h)), rt=3, l=9, t=3)

                currentDetect = np.array(
                    [
                        x1.detach().cpu(),
                        y1.detach().cpu(),
                        x2.detach().cpu(),
                        y2.detach().cpu(),
                        conf.detach().cpu(),
                    ],
                )
                detections = np.vstack((detections, currentDetect))
    resultTracker = tracker.update(detections)

    for result in resultTracker:
        x1, y1, x2, y2, id = result  # Tensors value of box
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)

        cvzone.putTextRect(frame, f"Count:{Counter["totalCount"]}", (50, 50)),
        
        cvzone.putTextRect(
            frame,
            f" {id}",
            (max(0, x1), max(35, y1)),
            scale=1,
            thickness=2,
            colorB=(0, 255, 0),
            offset=3,
        )
        if id not in Counter["id"]:
            Counter["totalCount"] += 1
            Counter["id"].append(id)
    cv2.imshow("Image", frame)
    if cv2.waitKey(1) and 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
