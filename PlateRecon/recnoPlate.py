# import cv2
# import torch
# import easyocr
# import numpy as np
#
# # Cargar modelo YOLOv5 personalizado (usa 'yolov5s.pt' o 'best.pt')
# model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')
#
# # Inicializa OCR
# reader = easyocr.Reader(['en'])
#
# # Iniciar captura de cámara
# cap = cv2.VideoCapture(0)
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Convertir a formato compatible con YOLO
#     results = model(frame)
#
#     # Extraer resultados
#     for *box, conf, cls in results.xyxy[0]:  # Cada detección
#         x1, y1, x2, y2 = map(int, box)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
#         # Recortar región de matrícula
#         plate_crop = frame[y1:y2, x1:x2]
#         if plate_crop.size == 0:
#             continue
#
#         # OCR
#         result = reader.readtext(plate_crop)
#         for detection in result:
#             text = detection[1]
#             print(f"[INFO] Matrícula detectada: {text}")
#             cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#
#     # Mostrar frame
#     cv2.imshow('YOLOv5 + OCR', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
