# ARマーカー(ID=0)を原点として、ファントムの位置をマーカー座標系で出す
# ZED2で撮影し、深度もZED2の実測値を使う
import pyzed.sl as sl
import os
import cv2
import numpy as np
from ultralytics import YOLO

# --- 設定項目 ---
# Yoloのrunsの中のptファイルを呼び出す
# このファイルがあるフォルダからの相対パス
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = YOLO(os.path.join(BASE_DIR, "runs", "detect", "train-8", "weights", "best.pt"))

# 使うARマーカー
ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_ID = 0
# マーカーの一辺(m)。距離は深度で出すので仮の値でOK(軸の向きと描画の長さにだけ使う)
MARKER_SIZE = 0.05

# ZED2を開く（ZED2は640x480が無いのでHD720 = 1280x720）
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.camera_fps = 30
init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
init_params.coordinate_units = sl.UNIT.METER
# OpenCV(solvePnP)と同じ座標系（X:右, Y:下, Z:前）
init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    raise RuntimeError("ZED2を開けませんでした")

runtime_params = sl.RuntimeParameters()
image_zed = sl.Mat()
depth_zed = sl.Mat()

# 内部パラメータ: ZED2の左カメラのキャリブレーション値
calib = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
FX, FY = calib.fx, calib.fy
PPX, PPY = calib.cx, calib.cy
WIDTH, HEIGHT = calib.image_size.width, calib.image_size.height

camera_matrix = np.array([[FX, 0, PPX],
                          [0, FY, PPY],
                          [0, 0, 1]], dtype=np.float64)
# ZEDの画像は歪み補正済み
dist_coeffs = np.zeros(5)

# ARマーカー検出器
aruco_detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT),
                                         cv2.aruco.DetectorParameters())

# マーカーの四隅のマーカー座標系での位置（左上, 右上, 右下, 左下）
half = MARKER_SIZE / 2
marker_obj_pts = np.array([[-half,  half, 0],
                           [ half,  half, 0],
                           [ half, -half, 0],
                           [-half, -half, 0]], dtype=np.float64)


def get_3d_pt(px, py):
    # ピクセル座標と深度から、カメラ座標系の3D座標(m)を返す
    # RealSenseのrs2_deproject_pixel_to_pointと同じ計算（歪みなし）
    # 周辺5x5の平均深度を取得して安定させる（eval.pyと同じ）
    px, py = int(round(px)), int(round(py))
    d_list = []
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            nx, ny = px + dx, py + dy
            if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                d = depth_image[ny, nx]
                if np.isfinite(d) and d > 0: d_list.append(d)
    if not d_list: return None
    d = np.mean(d_list)
    return np.array([(px - PPX) * d / FX, (py - PPY) * d / FY, d])


# ウィンドウの設定：WINDOW_NORMALにすることで、拡大縮小可能に
window_name = "Phantom Position from ArUco (Resizable)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

print("マーカー基準の位置検出モード: 開始 ('q'キーで終了)")

try:
    while True:
        if zed.grab(runtime_params) > sl.ERROR_CODE.SUCCESS:
            continue
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
        color_image = cv2.cvtColor(image_zed.get_data(), cv2.COLOR_BGRA2BGR)
        depth_image = depth_zed.get_data()  # 単位はm、無効値はNaN/inf

        # --- 1. ARマーカーの検出 ---
        marker_R = None    # マーカーの回転（カメラ座標系→マーカーの軸）
        marker_org = None  # マーカー中心のカメラ座標(m)
        corners, ids, _ = aruco_detector.detectMarkers(color_image)
        if ids is not None and MARKER_ID in ids.flatten():
            idx = list(ids.flatten()).index(MARKER_ID)
            img_pts = corners[idx].reshape(4, 2).astype(np.float64)
            cv2.aruco.drawDetectedMarkers(color_image, [corners[idx]], np.array([[MARKER_ID]]))

            # 軸の向きは四隅の見え方から求める
            ok, rvec, tvec = cv2.solvePnP(marker_obj_pts, img_pts, camera_matrix, dist_coeffs,
                                          flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvec, tvec, MARKER_SIZE)

                # 原点の位置はマーカー中心の深度から求める（深度が取れなければ見つからない扱い）
                mcx, mcy = img_pts.mean(axis=0)
                marker_org = get_3d_pt(mcx, mcy)
                if marker_org is not None:
                    marker_R, _ = cv2.Rodrigues(rvec)

        if marker_R is None:
            cv2.putText(color_image, f"Marker ID {MARKER_ID}: not found", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # --- 2. ファントムの検出 ---
        # ZED2は広角でファントムが小さく写るので、640に縮めず1280のまま検出する
        results = model(color_image, imgsz=1280, verbose=False)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                cls_name = model.names[int(box.cls[0])]
                if cls_name == "ファントム":
                    cls_name = "Phantom"
                conf = float(box.conf[0])

                color = (0, 255, 0)
                cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)
                cv2.circle(color_image, (cx, cy), 4, (255, 0, 0), -1)
                cv2.putText(color_image, f"{cls_name} ({conf:.2f})", (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if marker_R is None:
                    continue

                # --- 3. マーカー座標系へ変換 ---
                pt_phantom = get_3d_pt(cx, cy)
                if pt_phantom is None:
                    continue
                # (P - M) をマーカーの軸で表す: R^T (P - M)、mm単位
                pos_mm = marker_R.T @ (pt_phantom - marker_org) * 1000
                dist_mm = np.linalg.norm(pos_mm)

                cv2.putText(color_image, f"X:{pos_mm[0]:.1f} Y:{pos_mm[1]:.1f} Z:{pos_mm[2]:.1f}mm",
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                print(f"{cls_name}: X={pos_mm[0]:.1f}mm Y={pos_mm[1]:.1f}mm Z={pos_mm[2]:.1f}mm "
                      f"(距離 {dist_mm:.1f}mm)")

        cv2.imshow(window_name, color_image)
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    zed.close()
    cv2.destroyAllWindows()
