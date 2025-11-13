from ultralytics import YOLO
import cv2
import cvzone
from Assets.sort import *
import numpy as np

mask = cv2.imread("Assets/mask.png")
model = YOLO("../YOLO_Weights/yolov8l.pt")
# Traking
# max_age= After lost how long shall we wait to check for the object
traker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

# x and y co-ordinates of the line
limit = [300, 350, 690, 350]

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

cap = cv2.VideoCapture("Assets/videos/cars.mp4")

# Set height and width of the screen
[
    cap.set(p, v)
    for p, v in [(cv2.CAP_PROP_FRAME_WIDTH, 640), (cv2.CAP_PROP_FRAME_HEIGHT, 480)]
]

Counter = {"id": [], "totalCount": 0}

while True:
    # Getting image from video->success:Bool, img:frame
    success, frame = cap.read()
    img_region = cv2.bitwise_and(frame, mask)
    if not success:
        break

    results = model(img_region, stream=True)  # Passing img in the model
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = box.conf[0]  # Confidence
            cls_id = int(box.cls[0])  # Class Id
            current_class = classes[cls_id]

            x1, y1, x2, y2 = box.xyxy[0]  # Tensors value of box
            # converting to int the co-ordinates of box
            # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            h = y2 - y1
            w = x2 - x1
            if (
                current_class == "car"
                or current_class == "motorcycle"
                or current_class == "bus"
                or current_class == "truck"
            ):
                # cvzone.putTextRect(
                #     frame,
                #     f" {current_class} {conf:.2f}",
                #     (max(0, int(x1)), max(35, int(y1))),
                #     scale=0.9,
                #     thickness=1,
                #     offset=3,
                # )
                # Getting the rect on the object
                cvzone.cornerRect(
                    frame, (int(x1), int(y1), int(w), int(h)), l=9, t=3, rt=3
                )
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

    cv2.line(
        frame,
        (limit[0], limit[1]),
        (limit[2], limit[3]),
        color=(0, 0, 255),
        thickness=3,
    )
    resultTracker = traker.update(detections)

    for result in resultTracker:
        x1, y1, x2, y2, id = result  # Tensors value of box
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        h = y2 - y1
        w = x2 - x1

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # thickness of the text
        # padding between the text and the rectangle border
        cvzone.putTextRect(
            frame,
            f"{id}",
            (max(0, int(x1)), max(35, int(y1))),
            scale=1.5,
            thickness=2,
            offset=3,
        )
        cv2.circle(
            frame, center=(cx, cy), radius=5, color=(0, 0, 255), thickness=cv2.FILLED
        )
        cvzone.putTextRect(frame, f"Count:{Counter["totalCount"]}", (50, 50)),

        if limit[0] - 20 < cx < limit[2] + 20 and limit[1] - 20 < cy < limit[3] + 20:
            if id not in Counter["id"]:
                Counter["totalCount"] += 1
                Counter["id"].append(id)
                #Green color
                cv2.line(
                    frame,
                    (limit[0], limit[1]),
                    (limit[2], limit[3]),
                    color=(0, 255, 0),
                    thickness=3,
                )

    cv2.imshow("image", frame)
    # cv2.waitKey(0)
    if cv2.waitKey(1) and 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
