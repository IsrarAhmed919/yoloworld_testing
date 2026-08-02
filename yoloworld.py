from ultralytics import YOLOWorld
import cv2


cap, frame = cv2.VideoCapture(0), None
model = YOLOWorld("yolov8s-world.pt")
model.set_classes(["person", "headphones", "ring", "hand"])
cv2.namedWindow("YOLOv8 Inference", cv2.WINDOW_NORMAL)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, conf=0.40)
    annotated_frame = results[0].plot()
    cv2.resize(annotated_frame, (1024, 1024), interpolation=cv2.INTER_AREA)
    cv2.imshow("YOLOv8 Inference", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
