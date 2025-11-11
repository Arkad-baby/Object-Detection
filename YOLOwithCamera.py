from ultralytics import YOLO
import cv2
import cvzone

model=YOLO(model="../YOLO_Weights/yolov8n.pt") #version 8 nano

cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

while True:
    #ret= was the frame successfully captured? True/False
    ret,frame=cap.read()
    if not ret:
        break

    results=model(frame,stream=True)

    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            # x1,y1,h,w=box.xyhw[0]
            x1, y1, x2, y2 = int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())
            # x1, y1, h, w = int(x1.item()), int(y1.item()), int(h.item()), int(w.item())
            # cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),3)
            h=y2-y1
            w=x2-x1
            print(x1,y1,h,w)
            cvzone.cornerRect(frame,(x1,y1,w,h))

    cv2.imshow("Image", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cap.destroyAllWindows()