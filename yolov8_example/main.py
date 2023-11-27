from ultralytics import YOLO
import cv2
import time
import base64
from ultralytics.utils.plotting import Annotator

# Inisialisasi model YOLO
model = YOLO("yolov8n.pt")

video_path = "tes.mp4"
cap = cv2.VideoCapture(video_path)
pre_timeframe = 0
new_timeframe = 0


while True:
    # Baca frame dari video
    ret, frame = cap.read()

    # Hentikan program jika video sudah habis
    if not ret:
        break

    new_timeframe = time.time()
    fps = 1 / (new_timeframe - pre_timeframe)
    pre_timeframe = new_timeframe
    fps = int(fps)

    # Prediksi dengan model YOLO
    results = model.predict(source=frame)

    # Inisialisasi variabel untuk menghitung jumlah sepeda motor
    motorcycle_count = 0

    for r in results:
        annotator = Annotator(frame)

        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[
                0
            ]  # dapatkan koordinat kotak dalam format (top, left, bottom, right)
            c = box.cls

            print(model.names[int(c)])

            if model.names[int(c)] == "motorcycle":
                # Tambahkan jumlah sepeda motor
                motorcycle_count += 1

                label = "{}: {}".format(model.names[int(c)], format(box.conf[0], ".2f"))
                annotator.box_label(
                    b,
                    label,
                    color=(255, 0, 255),
                )

        frame = annotator.result()

    (h, w) = frame.shape[:2]
    # tambah fps ke window
    cv2.putText(
        frame,
        f"FPS: {fps}",
        org=(w - 132, 30),
        color=(0, 255, 0),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
    )
    cv2.putText(
        frame,
        f"Jumlah Sepeda Motor: {motorcycle_count}",
        (10, 30),  # Koordinat teks di dalam gambar (x, y)
        cv2.FONT_HERSHEY_SIMPLEX,
        1,  # Skala teks
        (255, 0, 255),  # Warna teks (dalam format BGR)
        2,  # Ketebalan teks
        cv2.LINE_AA,
    )

    # Tampilkan frame hasil
    cv2.imshow("Hasil", frame)

    # Keluar dari loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
