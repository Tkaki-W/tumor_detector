#写真の保存をimagesの中にする
import pyzed.sl as sl
import cv2
import numpy as np
import os

import glob
import re

# 保存フォルダ作成
save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "images")
os.makedirs(save_dir, exist_ok=True)

# 既存のファイルから最大のインデックスを取得
existing_files = glob.glob(os.path.join(save_dir, "img_*.jpg"))
if existing_files:
    indices = [int(re.search(r'img_(\d+)\.jpg', f).group(1)) for f in existing_files if re.search(r'img_(\d+)\.jpg', f)]
    i = max(indices) + 1 if indices else 0
else:
    i = 0

zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.camera_fps = 30
# 撮影だけなので深度計算はしない
init_params.depth_mode = sl.DEPTH_MODE.NONE
if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    raise RuntimeError("ZED2を開けませんでした")

runtime_params = sl.RuntimeParameters()
image_zed = sl.Mat()

print("sキー: 撮影保存 / qキー: 終了")

while True:
    if zed.grab(runtime_params) > sl.ERROR_CODE.SUCCESS:
        continue
    # 左カメラの画像（BGRA）をBGRに変換
    zed.retrieve_image(image_zed, sl.VIEW.LEFT)
    color = cv2.cvtColor(image_zed.get_data(), cv2.COLOR_BGRA2BGR)
    cv2.imshow("capture", color)

    key = cv2.waitKey(1)
    if key == ord('s'):
        path = f"{save_dir}/img_{i:04d}.jpg"
        cv2.imwrite(path, color)
        print(f"保存: {path}")
        i += 1
    elif key == ord('q'):
        break

zed.close()
cv2.destroyAllWindows()
