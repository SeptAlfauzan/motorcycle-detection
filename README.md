## Motorbike object detection using Yolov8 and Yolox

|                              Yolox                              |                              YoloV8                              |
| :-------------------------------------------------------------: | :--------------------------------------------------------------: |
| ![Yolox output](./outputs/Screenshot%202023-11-24%20230726.png) | ![YoloV8 output](./outputs/Screenshot%202023-11-24%20230125.png) |

This program is used to count how many motorbike in current frame by detect the motorbike object

How to run

1. Create virtual environtment

   ```
   python -m venv <virtual env name>

   ```

2. Install required pip packages
   ```
   pip install -r requirements.txt
   ```
3. To run YoloV8 version, run this command
   ```
   python ./yolov8_example/main.py
   ```
4. To run Yolox version, run this command
   ```
   python ./yolox_example/main.py
   ```
5. For Yolox version, to detect object from image file run this command
   ```
   python .\yolox_example\main.py --input <image file dir> -v
   ``
   ```
