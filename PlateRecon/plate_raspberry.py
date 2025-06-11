import os
import warnings
import time
import cv2
import easyocr
import torch
import numpy as np
import csv
import re
from datetime import datetime
from collections import defaultdict
from flask import Flask, Response, jsonify

# === Configuración inicial ===
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PLATE_DIR = os.path.join(ROOT_DIR, 'plates_detected')
FRAME_DIR = os.path.join(ROOT_DIR, 'frames_detected')
CSV_LOG_PATH = os.path.join(ROOT_DIR, 'detections_log.csv')
TORCHSCRIPT_MODEL_PATH = os.path.join(ROOT_DIR, 'best.torchscript.pt')  # ruta relativa o absoluta

os.makedirs(PLATE_DIR, exist_ok=True)
os.makedirs(FRAME_DIR, exist_ok=True)

VALID_PLATE_REGEX = re.compile(r'^\d{3,4}\s?[A-Z]{3}$')
IGNORED_WORDS = {"BOLIVIA", "TRANSPORTE", "VEHICULO"}
plate_counter = defaultdict(int)
last_seen = {}

# Crear CSV si no existe
if not os.path.exists(CSV_LOG_PATH):
    with open(CSV_LOG_PATH, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Fecha', 'Hora', 'Matrícula', 'Cropped Path', 'Frame Path'])

# === Inicialización de cámara ===
def init_camera():
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        print("✅ Cámara inicializada correctamente.")
        return cap
    else:
        print("❌ No se pudo acceder a la cámara.")
        return None

cap = init_camera()

def create_black_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)

# === Modelo TorchScript y OCR ===
print(f"🔍 Cargando modelo TorchScript desde {TORCHSCRIPT_MODEL_PATH}")
model = torch.jit.load(TORCHSCRIPT_MODEL_PATH)
model.eval()
reader = easyocr.Reader(['en'])

CONFIDENCE_THRESHOLD = 0.4

def inference_torchscript(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    img = torch.tensor(img, dtype=torch.float)

    with torch.no_grad():
        pred = model(img)[0]

    pred = pred.cpu()
    boxes = []

    for det in pred:
        x1, y1, x2, y2, conf, cls = det.tolist()
        if conf >= CONFIDENCE_THRESHOLD:
            boxes.append((int(x1), int(y1), int(x2), int(y2), conf, cls))

    return boxes

# === Flask App ===
app = Flask(__name__)
last_plate = ""
last_detection_time = 0

def gen_frames():
    global last_plate, last_detection_time
    frame_rate = 15
    prev_time = 0

    while True:
        try:
            time_elapsed = time.time() - prev_time
            if time_elapsed < 1. / frame_rate:
                time.sleep(0.01)
                continue
            prev_time = time.time()

            if cap is None:
                frame = create_black_frame()
            else:
                success, frame = cap.read()
                if not success:
                    frame = create_black_frame()

            detections_yolo = inference_torchscript(frame)

            for x1, y1, x2, y2, conf, cls in detections_yolo:
                cropped = frame[y1:y2, x1:x2]
                if cropped.size == 0:
                    continue

                detections = reader.readtext(cropped)
                for detection in detections:
                    text = detection[1].strip().upper().replace("-", " ")

                    if text in IGNORED_WORDS or not VALID_PLATE_REGEX.match(text):
                        continue

                    if text in last_seen and time.time() - last_seen[text] < 30:
                        continue

                    if time.time() - last_detection_time < 3:
                        continue

                    last_plate = text
                    last_detection_time = time.time()
                    last_seen[text] = time.time()

                    timestamp = datetime.now()
                    timestamp_str = timestamp.strftime('%Y-%m-%d_%H-%M-%S')
                    date_str = timestamp.strftime('%Y-%m-%d')
                    time_str = timestamp.strftime('%H:%M:%S')

                    plate_counter[text] += 1

                    cropped_filename = f"{timestamp_str}_{text}_{plate_counter[text]}.jpg"
                    cropped_path = os.path.join(PLATE_DIR, cropped_filename)
                    cv2.imwrite(cropped_path, cropped)

                    full_filename = f"{timestamp_str}_{text}_{plate_counter[text]}_full.jpg"
                    full_path = os.path.join(FRAME_DIR, full_filename)
                    cv2.imwrite(full_path, frame)

                    with open(CSV_LOG_PATH, mode='a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([date_str, time_str, text, cropped_path, full_path])

                    print("========================================")
                    print(f"✅ MATRÍCULA DETECTADA: {text}")
                    print(f"🕒 HORA: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"🖼️ ARCHIVO CROPPED: {cropped_filename}")
                    print(f"🖼️ ARCHIVO FRAME  : {full_filename}")
                    print("========================================\n")

            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        except Exception as e:
            print(f"Error en gen_frames: {str(e)}")
            time.sleep(0.1)

@app.route('/video')
def video():
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
    )

@app.route('/last_plate')
def get_last_plate():
    if time.time() - last_detection_time < 30:
        return jsonify({'plate': last_plate})
    return jsonify({'plate': ''})

if __name__ == '__main__':
    try:
        print("🚀 Iniciando servidor Flask en http://0.0.0.0:5000")
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
