import pyzed.sl as sl
import os
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 設定 ---
# このファイルがあるフォルダからの相対パス
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "runs", "detect", "train-8", "weights", "best.pt")
model = YOLO(MODEL_PATH)

# ZED2初期化（ZED2は640x480が無いのでHD720 = 1280x720）
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.camera_fps = 30
init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
init_params.coordinate_units = sl.UNIT.METER
# RealSenseと同じ座標系（X:右, Y:下, Z:前）
init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    raise RuntimeError("ZED2を開けませんでした")

# ZEDの深度は最初から左カメラ画像に合っているので、alignは不要
intrinsics = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
img_w = intrinsics.image_size.width
img_h = intrinsics.image_size.height
runtime_params = sl.RuntimeParameters()
image_zed = sl.Mat()
depth_zed = sl.Mat()

# ピクセル座標から3D空間座標(x, y, z)に変換（rs2_deproject_pixel_to_pointの代わり）
# ZEDの画像は歪み補正済みなのでピンホールモデルそのまま
def deproject_pixel_to_point(intrinsics, pixel, d):
    px, py = pixel
    return [(px - intrinsics.cx) / intrinsics.fx * d, (py - intrinsics.cy) / intrinsics.fy * d, d]

def show_3d_tumor(depth_roi, intrinsics, x_offset, y_offset):
    """
    指定された範囲の深度データを3Dプロットする
    """
    h, w = depth_roi.shape
    # 間引き（重くなるのを防ぐため）
    step = 2

    # 3D座標リスト
    xs, ys, zs = [], [], []

    for v in range(0, h, step):
        for u in range(0, w, step):
            d = depth_roi[v, u]
            # ZEDの無効値(NaN/inf)を除外
            if np.isfinite(d) and d > 0:
                # 3D座標に変換
                pt = deproject_pixel_to_point(intrinsics, [u + x_offset, v + y_offset], d)
                xs.append(pt[0] * 1000) # mm
                ys.append(pt[1] * 1000) # mm
                zs.append(pt[2] * 1000) # mm

    if not zs: return

    # グラフ作成
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 表面の高さ（深度）を反転させて「盛り上がり」として見せる
    # 一番深いところ（底）を0基準にする
    max_z = max(zs)
    zs_rel = [max_z - z for z in zs]

    sc = ax.scatter(xs, ys, zs_rel, c=zs_rel, cmap='jet', s=2)
    plt.colorbar(sc, label='Height (mm)')

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Height (mm)')
    ax.set_title('Tumor 3D Surface')

    # 視点設定
    ax.view_init(elev=30, azim=45)
    plt.show()

print("3D可視化モード: 開始")
print("'s'キー: 検出中の腫瘍を3D表示")
print("'q'キー: 終了")

try:
    while True:
        if zed.grab(runtime_params) > sl.ERROR_CODE.SUCCESS:
            continue
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)

        color_image = cv2.cvtColor(image_zed.get_data(), cv2.COLOR_BGRA2BGR)
        depth_image = depth_zed.get_data()

        # ZED2は広角でファントムが小さく写るので、640に縮めず1280のまま検出する
        results = model(color_image, imgsz=1280, verbose=False)

        current_box = None
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                current_box = (x1, y1, x2, y2)
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(color_image, "Ready to Scan (Press 's')", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("3D Visualization Waiter", color_image)

        key = cv2.waitKey(1)
        if key == ord('s') and current_box:
            x1, y1, x2, y2 = current_box
            # 枠を少し広めに取得
            margin = 10
            rx1 = max(0, x1 - margin)
            ry1 = max(0, y1 - margin)
            rx2 = min(img_w, x2 + margin)
            ry2 = min(img_h, y2 + margin)

            # 深度ROIを取得
            depth_roi = depth_image[ry1:ry2, rx1:rx2]
            # ZEDの深度は最初からメートル単位(float32)なので単位変換は不要
            depth_roi_meters = depth_roi.astype(float)

            print("3Dモデル生成中...")
            show_3d_tumor(depth_roi_meters, intrinsics, rx1, ry1)

        elif key == ord('q'):
            break

finally:
    zed.close()
    cv2.destroyAllWindows()
