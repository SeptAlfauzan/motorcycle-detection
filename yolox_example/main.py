from yolox import YoloX
import cv2 as cv
import numpy as np
import argparse
from sort import Sort

classes = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


totalCount = []
tracker = Sort(max_age=100)
tracking = False


def letterbox(srcimg, target_size=(640, 640)):
    padded_img = np.ones((target_size[0], target_size[1], 3)).astype(np.float32) * 114.0
    ratio = min(target_size[0] / srcimg.shape[0], target_size[1] / srcimg.shape[1])
    resized_img = cv.resize(
        srcimg,
        (int(srcimg.shape[1] * ratio), int(srcimg.shape[0] * ratio)),
        interpolation=cv.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[
        : int(srcimg.shape[0] * ratio), : int(srcimg.shape[1] * ratio)
    ] = resized_img

    return padded_img, ratio


def unletterbox(bbox, letterbox_scale):
    return bbox / letterbox_scale


def vis(
    dets,
    srcimg,
    letterbox_scale,
    tracking,
    confidence,
    fps=None,
):
    res_img = srcimg.copy()

    if fps is not None:
        fps_label = "FPS: %.2f" % fps
        cv.putText(
            res_img, fps_label, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )

    # Initialize counter for motorcycles
    motorcycle_count = 0

    detections = np.empty((0, 5))
    for det in dets:
        box = unletterbox(det[:4], letterbox_scale).astype(np.int32)
        score = det[-2]
        cls_id = int(det[-1])

        x0, y0, x1, y1 = box

        if classes[cls_id] == "motorcycle":
            currentArray = np.array([x0, y0, x1, y1, score])

            detections = np.vstack((detections, currentArray))

            print(detections)
            # Increment motorcycle count
            motorcycle_count += 1

            text = "{}:{:.1f}%".format(classes[cls_id], score * 100)
            font = cv.FONT_HERSHEY_SIMPLEX
            txt_size = cv.getTextSize(text, font, 0.4, 1)[0]
            cv.rectangle(res_img, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv.rectangle(
                res_img,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                (255, 255, 255),
                -1,
            )
            cv.putText(
                res_img, text, (x0, y0 + txt_size[1]), font, 0.4, (0, 0, 0), thickness=1
            )
    resultsTracker = tracker.update(detections)

    # display id
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

        cv.putText(
            res_img,
            id_txt,
            org=(cx, cy),
            fontFace=cv.FONT_HERSHEY_SIMPLEX,
            fontScale=1.4,
            thickness=3,
            color=(0, 255, 255),
        )

    # Display motorcycle count on the frame
    if tracking == False:
        cv.putText(
            res_img,
            f"Jumlah Sepeda Motor: {motorcycle_count}",
            (10, 30),  # Koordinat teks di dalam gambar (x, y)
            cv.FONT_HERSHEY_SIMPLEX,
            1,  # Skala teks
            (255, 0, 255),  # Warna teks (dalam format BGR)
            2,  # Ketebalan teks
            cv.LINE_AA,
        )
    else:
        cv.putText(
            res_img,
            f"Total Sepeda Motor: {len(totalCount)}",
            (10, 30),  # Koordinat teks di dalam gambar (x, y)
            cv.FONT_HERSHEY_SIMPLEX,
            1,  # Skala teks
            (255, 0, 255),  # Warna teks (dalam format BGR)
            2,  # Ketebalan teks
            cv.LINE_AA,
        )

    return res_img, motorcycle_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YoloX inference using OpenCV")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Path to the input video. Omit for using default camera.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="./yolox_example/object_detection_yolox_2022nov.onnx",
        help="Path to the model",
    )
    parser.add_argument(
        "--confidence", default=0.5, type=float, help="Class confidence"
    )
    parser.add_argument(
        "--nms", default=0.5, type=float, help="Enter nms IOU threshold"
    )
    parser.add_argument("--obj", default=0.5, type=float, help="Enter object threshold")
    parser.add_argument(
        "--vis",
        "-v",
        action="store_true",
        help="Specify to open a window for result visualization. This flag is invalid when using camera.",
    )
    parser.add_argument(
        "--tracking",
        "-t",
        default=False,
        type=bool,
        help="Value to display total count tracking of motorcycle",
    )

    args = parser.parse_args()

    model_net = YoloX(
        modelPath=args.model,
        confThreshold=args.confidence,
        nmsThreshold=args.nms,
        objThreshold=args.obj,
        backendId=cv.dnn.DNN_BACKEND_CUDA,
        targetId=cv.dnn.DNN_TARGET_CUDA,
    )

    if args.input is not None:
        cap = cv.VideoCapture(args.input)
    else:
        print("Press any key to stop video capture")
        cap = cv.VideoCapture(0)

    if args.tracking is True:
        tracking = True

    confidence = 0.5
    if args.confidence is not None:
        confidence = args.confidence

    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print("No frames grabbed!")
            break

        frame = cv.resize(frame, (960, 540))
        input_blob = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        input_blob, letterbox_scale = letterbox(input_blob)

        # Inference
        preds = model_net.infer(input_blob)

        # Visualization
        result_img, motorcycle_count = vis(
            preds, frame, letterbox_scale, confidence=confidence, tracking=tracking
        )

        cv.imshow("YoloX Demo", result_img)

    # Release video capture and close window
    cap.release()
    cv.destroyAllWindows()
