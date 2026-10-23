#YOLOの学習を行う
import os
from ultralytics import YOLO

if __name__ == '__main__':
    # このファイルがあるフォルダからの相対パス（学習結果もこのフォルダのruns/detectに保存）
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model = YOLO(os.path.join(BASE_DIR, 'yolov8n.pt'))
    model.train(data=os.path.join(BASE_DIR, 'dataset.yaml'), epochs=100, imgsz=640,
                project=os.path.join(BASE_DIR, 'runs', 'detect'))