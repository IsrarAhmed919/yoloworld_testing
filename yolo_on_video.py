from ultralytics import YOLO
import cv2


cap, frame = cv2.VideoCapture(0), None
model = YOLO("yolov8n.pt")
cv2.namedWindow("YOLOv8 Inference", cv2.WINDOW_NORMAL)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Inference", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
