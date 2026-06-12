#ptファイルを呼び出して対象をくくる
import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO

#Yoloのrunsの中のptファイルを呼び出す
model = YOLO(r"C:\Users\mswas\Desktop\オクターブン卒論\Yolo操作\runs\detect\train-8\weights\best.pt")

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

while True:
    frames = pipeline.wait_for_frames()
    color = np.asanyarray(frames.get_color_frame().get_data())
    
    results = model(color)
    annotated = results[0].plot()
    
    cv2.imshow("YOLOv8 Detection", annotated)
    if cv2.waitKey(1) == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()