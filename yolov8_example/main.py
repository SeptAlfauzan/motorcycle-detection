import numpy as np
from ultralytics import YOLO
import cv2
import time
import base64
from ultralytics.utils.plotting import Annotator
import argparse

from sort import Sort

# Initialize object tracker
tracker = Sort(max_age=100)

# Inisialisasi model YOLO
model = YOLO("yolov8n.pt")

video_path = "./videos/4.mp4"

parser = argparse.ArgumentParser(description="YoloX inference using OpenCV")
parser.add_argument(
    "--input",
    "-i",
    type=str,
    help="Path to the input video. Omit for using default camera.",
)
parser.add_argument(
    "--tracking",
    "-t",
    default=False,
    type=bool,
    help="Value to display total count tracking of motorcycle",
)

parser.add_argument(
    "--confidence",
    "-c",
    type=float,
    help="Result confidence of motorcycle object that will use as threshold to show",
)

args = parser.parse_args()
confidence = 0.5

if args.input is not None:
    cap = cv2.VideoCapture(args.input)
else:
    cap = cv2.VideoCapture(video_path)

if args.confidence is not None:
    confidence = args.confidence

pre_timeframe = 0
new_timeframe = 0

totalCount = []

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
        detections = np.empty((0, 5))

        annotator = Annotator(frame)

        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[
                0
            ]  # dapatkan koordinat kotak dalam format (top, left, bottom, right)
            c = box.cls

            if model.names[int(c)] == "motorcycle" and box.conf[0] > confidence:
                score = box.conf.item() * 100
                class_id = int(box.cls.item())
                x1, y1, x2, y2 = np.squeeze(box.xyxy.numpy()).astype(int)

                currentArray = np.array([x1, y1, x2, y2, score])

                detections = np.vstack((detections, currentArray))
                # Tambahkan jumlah sepeda motor
                motorcycle_count += 1

                label = "{}: {}".format(model.names[int(c)], format(box.conf[0], ".2f"))
                annotator.box_label(
                    b,
                    label,
                    color=(255, 0, 255),
                )
        resultsTracker = tracker.update(detections)

        frame = annotator.result()

        for result in resultsTracker:
            # Get the tracker results
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
            # print(result)

            # Display current objects IDs
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2
            id_txt = f"ID: {str(id)}"

            if id not in totalCount:
                totalCount.append(id)

            cv2.putText(
                frame,
                id_txt,
                org=(cx, cy),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.4,
                thickness=3,
                color=(0, 255, 255),
            )

        # print(id)
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
    print(args.tracking)
    if args.tracking is False:
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
    else:
        cv2.putText(
            frame,
            f"Total Sepeda Motor: {len(totalCount)}",
            (10, 30),  # Koordinat teks di dalam gambar (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,
            1,  # Skala teks
            (255, 0, 255),  # Warna teks (dalam format BGR)
            2,  # Ketebalan teks
            cv2.LINE_AA,
        )

    frame = cv2.resize(frame, (960, 540))

    # Tampilkan frame hasil
    cv2.imshow("Hasil", frame)

    # Keluar dari loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
