#YOLOの学習を行う
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    model.train(data=r'C:\Users\mswas\Desktop\オクターブン卒論\Yolo操作\dataset.yaml', epochs=100, imgsz=640)