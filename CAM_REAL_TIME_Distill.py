import cv2
from ultralytics import YOLO
import numpy as np
import re
import os
from datetime import datetime
from paddleocr import PaddleOCR

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

cap = cv2.VideoCapture(0)
model = YOLO("best.pt")
ocr = PaddleOCR(use_angle_cls=True, use_gpu=True)

className = ["License"]

def detect_plate_type(x1, y1, x2, y2):
    width = x2 - x1
    height = y2 - y1
    return "rectangle" if width / height > 3 else "square"

def split_license_plate(frame, x1, y1, x2, y2):
    plate_img = frame[y1:y2, x1:x2]
    height = y2 - y1
    line1 = plate_img[0:height // 2, :]
    line2 = plate_img[height // 2:, :]
    return line1, line2

def paddle_ocr_line(img):
    result = ocr.ocr(img, det=False, rec=True, cls=False)
    text = ""
    for r in result:
        scores = int(r[0][1] * 100) if not np.isnan(r[0][1]) else 0
        if scores > 60:
            text = r[0][0]
    return re.sub('[\W]', '', text).replace("O", "0").replace("粤", "")

def process_license_plate(frame, x1, y1, x2, y2):
    plate_type = detect_plate_type(x1, y1, x2, y2)
    if plate_type == "rectangle":
        return paddle_ocr_line(frame[y1:y2, x1:x2])
    elif plate_type == "square":
        line1, line2 = split_license_plate(frame, x1, y1, x2, y2)
        return paddle_ocr_line(line1) + paddle_ocr_line(line2)
    return ""

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = process_license_plate(frame, x1, y1, x2, y2)
        if label:
            print(f"Detected License Plate: {label}")
            textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
            c2 = x1 + textSize[0], y1 - textSize[1] - 3
            cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
    
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
 