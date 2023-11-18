from ultralytics import YOLO
import cv2
import base64
from ultralytics.utils.plotting import (
    Annotator,
)

model = YOLO("yolov8n.pt")
img = cv2.imread("./vehicles.jpg")

results = model.predict(source=img)
# images = results
print(results)

for r in results:
    annotator = Annotator(img)
    boxes = r.boxes
    for box in boxes:
        b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
        c = box.cls

        print(model.names[int(c)])

        if model.names[int(c)] == "motorcycle":
            label = "{}: {}".format(model.names[int(c)], format(box.conf[0], ".2f"))
            annotator.box_label(
                b,
                label,
                color=(255, 0, 255),
            )

img = annotator.result()
# convert to base64
string = base64.b64encode(cv2.imencode(".jpg", img)[1]).decode()
print(string)
# write base64 string image to file.txt
text_file = open("base64.txt", "w")
text_file.write("base64: %s" % string)
text_file.close()

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
