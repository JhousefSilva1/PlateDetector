# import cv2
# import numpy as np
# import easyocr
#
# # Inicializa el lector OCR de EasyOCR con idioma en inglés (puedes cambiar)
# reader = easyocr.Reader(['en'], gpu=False)
#
# # Inicia la captura de video (0 para cámara por defecto)
# cap = cv2.VideoCapture(0)
#
# print("[INFO] Presiona 'q' para salir")
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Redimensiona para agilizar el proceso
#     frame_resized = cv2.resize(frame, (640, 480))
#
#     # Convertimos a escala de grises
#     gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
#
#     # Aplicamos un filtro bilateral para suavizar sin perder bordes
#     filtered = cv2.bilateralFilter(gray, 11, 17, 17)
#
#     # Detectamos bordes con Canny
#     edged = cv2.Canny(filtered, 30, 200)
#
#     # Encontramos contornos
#     contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
#
#     plate_img = None
#     for c in contours:
#         approx = cv2.approxPolyDP(c, 0.018 * cv2.arcLength(c, True), True)
#         if len(approx) == 4:
#             x, y, w, h = cv2.boundingRect(approx)
#             plate_img = frame_resized[y:y + h, x:x + w]
#             cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             break
#
#     # Si se encontró una posible placa
#     if plate_img is not None:
#         # OCR con EasyOCR
#         result = reader.readtext(plate_img)
#         for detection in result:
#             text = detection[1]
#             print(f"[INFO] Texto detectado: {text}")
#             cv2.putText(frame_resized, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
#
#     # Muestra el frame
#     cv2.imshow("Reconocimiento de Matrícula", frame_resized)
#
#     # Salir con 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Limpieza
# cap.release()
# cv2.destroyAllWindows()
