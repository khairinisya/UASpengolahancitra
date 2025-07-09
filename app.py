
from flask import Flask, render_template, request, Response
from ultralytics import YOLO
import os
import uuid
import shutil
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Pastikan folder uploads ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model YOLO12n sekali saja
model = YOLO("yolo12n.pt")  # otomatis download jika belum ada

# --- Route halaman utama upload gambar ---
@app.route('/')
def index():
    return render_template('index.html')

# --- Route deteksi gambar upload ---
@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return 'No file uploaded', 400

    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400

    # Simpan file gambar dengan nama unik
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Jalankan deteksi objek menggunakan YOLO
    results = model.predict(source=filepath, save=True, project='runs/detect', name='predict', exist_ok=True)

    # Ambil path hasil deteksi yang disimpan oleh YOLO
    result_dir = results[0].save_dir
    detected_image_path = os.path.join(result_dir, filename)
    final_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Pindahkan hasil deteksi ke folder static/uploads supaya bisa diakses lewat web
    if os.path.exists(detected_image_path):
        shutil.move(detected_image_path, final_image_path)

    # Render halaman hasil deteksi dengan gambar output
    return render_template('result.html', result_image=filename)

# --- Fungsi generator stream kamera dengan deteksi YOLOv12 ---
def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Tidak dapat membuka webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Jalankan deteksi objek pada frame kamera
        results = model(frame)

        # Gambar bounding box hasil deteksi ke frame
        frame = results[0].plot()

        # Encode frame menjadi JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Kirim frame sebagai multipart HTTP response untuk streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# --- Route streaming video kamera dengan deteksi ---
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Route halaman untuk streaming kamera ---
@app.route('/camera')
def camera():
    return render_template('camera.html')

if __name__ == '__main__':
    app.run(debug=True)