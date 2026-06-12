#写真の保存をimagesの中にする
import pyrealsense2 as rs
import cv2
import numpy as np
import os

import glob
import re

# 保存フォルダ作成
save_dir = "dataset/images"
os.makedirs(save_dir, exist_ok=True)

# 既存のファイルから最大のインデックスを取得
existing_files = glob.glob(os.path.join(save_dir, "img_*.jpg"))
if existing_files:
    indices = [int(re.search(r'img_(\d+)\.jpg', f).group(1)) for f in existing_files if re.search(r'img_(\d+)\.jpg', f)]
    i = max(indices) + 1 if indices else 0
else:
    i = 0

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

print("sキー: 撮影保存 / qキー: 終了")

while True:
    frames = pipeline.wait_for_frames()
    color = np.asanyarray(frames.get_color_frame().get_data())
    cv2.imshow("capture", color)
    
    key = cv2.waitKey(1)
    if key == ord('s'):
        path = f"{save_dir}/img_{i:04d}.jpg"
        cv2.imwrite(path, color)
        print(f"保存: {path}")
        i += 1
    elif key == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()