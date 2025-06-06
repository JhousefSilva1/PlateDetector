import cv2
import torch
import easyocr
import numpy as np

# Cargar modelo YOLOv5 entrenado desde el repo local
model = torch.hub.load('yolov5', 'custom', path='yolov5/runs/train/lp_detector3/weights/best.pt', source='local')

# Iniciar OCR
reader = easyocr.Reader(['en'])

# Iniciar cámara
cap = cv2.VideoCapture(0)

print("[INFO] Presiona 'q' para salir...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detección
    results = model(frame)

    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        cropped = frame[y1:y2, x1:x2]

        if cropped.size == 0:
            continue

        detections = reader.readtext(cropped)
        for detection in detections:
            text = detection[1]
            print(f"[INFO] Matrícula detectada: {text}")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)

    cv2.imshow("Reconocimiento de Matrículas", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
