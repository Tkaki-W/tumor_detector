# ptファイルを呼び出して対象をくくる
import pyzed.sl as sl
import os
import cv2
import numpy as np
from ultralytics import YOLO

# --- 設定項目 ---
# カメラの光軸（正面）と重力方向（鉛直下向き）のなす角度（度数法）逐一やってね
# カメラが真下を向いているなら 0、水平を向いているなら 90、斜め下45度なら 45
CAMERA_ANGLE_DEG = 44.0

# Yoloのrunsの中のptファイルを呼び出す
# このファイルがあるフォルダからの相対パス
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = YOLO(os.path.join(BASE_DIR, "runs", "detect", "train-8", "weights", "best.pt"))

zed = sl.Camera()
init_params = sl.InitParameters()

# カラーと深度の設定（ZED2は640x480が無いのでHD720 = 1280x720）
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.camera_fps = 30
init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
init_params.coordinate_units = sl.UNIT.METER
# RealSenseと同じ座標系（X:右, Y:下, Z:前）
init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE

# ストリーミング開始
if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    raise RuntimeError("ZED2を開けませんでした")

# カメラの内部パラメータ（焦点距離など）を取得して実寸計算に使用
intrinsics = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
fx = intrinsics.fx
fy = intrinsics.fy
img_w = intrinsics.image_size.width
img_h = intrinsics.image_size.height
cam_cx = img_w // 2
cam_cy = img_h // 2

# ZEDの深度は最初から左カメラ画像に合っているので、alignは不要
runtime_params = sl.RuntimeParameters()
image_zed = sl.Mat()
depth_zed = sl.Mat()

# ピクセル座標から3D空間座標(x, y, z)に変換（rs2_deproject_pixel_to_pointの代わり）
# ZEDの画像は歪み補正済みなのでピンホールモデルそのまま
def deproject_pixel_to_point(px, py, d):
    return [(px - intrinsics.cx) / fx * d, (py - intrinsics.cy) / fy * d, d]

# ウィンドウの設定：WINDOW_NORMALにすることで、拡大縮小可能に
window_name = "Tumor Detection (Resizable)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

print("腫瘍半径検知モード: 開始 ('q'キーで終了)")

try:
    while True:
        if zed.grab(runtime_params) > sl.ERROR_CODE.SUCCESS:
            continue
        #カラーと深度を別々に取り出す
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)

        color_image = cv2.cvtColor(image_zed.get_data(), cv2.COLOR_BGRA2BGR)
        depth_image = depth_zed.get_data()  # 単位はm、無効値はNaN/inf
        #機械学習モデルにかける！
        # ZED2は広角でファントムが小さく写るので、640に縮めず1280のまま検出する
        results = model(color_image, imgsz=1280, verbose=False)

        #検出分の数だけ繰り返す
        for result in results:
            for box in result.boxes:
                # 枠の座標 (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # --- 1. 3D座標の取得関数 ---
                def get_3d_pt(px, py):
                    # 周辺5x5の平均深度を取得して安定させる
                    d_list = []
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            nx, ny = px + dx, py + dy
                            if 0 <= nx < img_w and 0 <= ny < img_h:
                                d = depth_image[ny, nx]
                                if np.isfinite(d) and d > 0: d_list.append(d)
                    if not d_list: return None
                    avg_d = np.mean(d_list)
                    # ピクセル座標から3D空間座標(x, y, z)に変換
                    return deproject_pixel_to_point(px, py, avg_d), avg_d

                # 中心の3D座標（頂点）
                res_c = get_3d_pt(cx, cy)
                if not res_c: continue
                pt_center, dist_center = res_c

                # 枠の上下左右4点の3D座標（底辺/境界線）
                edge_pts = []
                for ex, ey in [(x1, cy), (x2, cy), (cx, y1), (cx, y2)]:
                    res_e = get_3d_pt(ex, ey)
                    if res_e: edge_pts.append(res_e[0])

                # --- 2. 半径(mm)の計算 (枠のサイズベース + fx使用) ---
                width_px = x2 - x1
                height_px = y2 - y1
                # 縦横平均からピクセル半径を算出
                radius_px = (width_px) / 2

                if dist_center > 0:
                    # ピクセルから実寸(mm)への変換公式: (px * 距離) / fx
                    # YOLOの枠が少し内側を通るため、1.15倍の補正をかけて調整
                    radius_mm = (radius_px * dist_center * 1000 / fx) * 1.4
                else:
                    radius_mm = 0

                # 写真中央（カメラ光軸）からのX方向、Y方向のズレ (実寸 mm)
                # pt_center (X_c, Y_c, Z_c) の X_c, Y_c は光軸からのメートル単位の距離
                dx_mm = pt_center[0] * 1000
                dy_mm = pt_center[1] * 1000

                # 鉛直方向の距離（高さ/深さ）の計算
                theta_rad = np.radians(CAMERA_ANGLE_DEG)
                # pt_center[1] はカメラ座標系のY(下方向)、pt_center[2] はカメラ座標系のZ(前方向)
                vertical_dist = pt_center[1] * np.sin(theta_rad) + pt_center[2] * np.cos(theta_rad)

                # ラベル作成
                cls = int(box.cls[0])
                cls_name = model.names[cls]
                if cls_name == "ファントム":
                    cls_name = "Phantom"
                conf = float(box.conf[0])
                line1 = f"{cls_name} ({conf:.2f})"
                line2 = f"R: {radius_mm:.1f}mm Depth: {dist_center:.2f}m"
                line3 = f"dx: {dx_mm:.1f}mm dy: {dy_mm:.1f}mm"
                line4 = f"Z(vert): {vertical_dist:.3f}m"

                # 描画
                color = (0, 255, 0)
                cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(color_image, line1, (x1, y1 - 59),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                """
                cv2.putText(color_image, line2, (x1, y1 - 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                """
                cv2.putText(color_image, line3, (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(color_image, line4, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # 中心点の可視化
                cv2.circle(color_image, (cx, cy), 4, (255, 0, 0), -1)

        # 画面中央（原点）にレティクル（赤い十字）を描画
        cv2.line(color_image, (cam_cx - 15, cam_cy), (cam_cx + 15, cam_cy), (0, 0, 255), 2)
        cv2.line(color_image, (cam_cx, cam_cy - 15), (cam_cx, cam_cy + 15), (0, 0, 255), 2)

        cv2.imshow(window_name, color_image)
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    zed.close()
    cv2.destroyAllWindows()
