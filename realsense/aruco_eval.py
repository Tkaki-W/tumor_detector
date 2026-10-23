# ARマーカー(ID=0)を原点として、ファントムの位置をマーカー座標系で出す
# ※ 今はRealSenseがないので、Webカメラで撮影し、深度は固定値で代用している
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

# Webカメラの番号（内蔵カメラなら大体0）
CAMERA_INDEX = 1
WIDTH, HEIGHT = 640, 480

# RealSenseの代わりの固定値
# 深度(m): カメラからマーカー・ファントムまでの距離。RealSenseが来たら実測値に置き換える
FIXED_DEPTH = 0.80
# 内部パラメータ: RealSense D435のカラー(640x480)のおおよその値
FX, FY = 615.0, 615.0
PPX, PPY = WIDTH / 2, HEIGHT / 2

camera_matrix = np.array([[FX, 0, PPX],
                          [0, FY, PPY],
                          [0, 0, 1]], dtype=np.float64)
dist_coeffs = np.zeros(5)

# Webカメラを開く
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
if not cap.isOpened():
    raise RuntimeError(f"カメラ {CAMERA_INDEX} を開けませんでした")

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
    d = FIXED_DEPTH
    return np.array([(px - PPX) * d / FX, (py - PPY) * d / FY, d])


# ウィンドウの設定：WINDOW_NORMALにすることで、拡大縮小可能に
window_name = "Phantom Position from ArUco (Resizable)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

print("マーカー基準の位置検出モード: 開始 ('q'キーで終了)")

try:
    while True:
        ok, color_image = cap.read()
        if not ok:
            continue

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
                marker_R, _ = cv2.Rodrigues(rvec)
                cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvec, tvec, MARKER_SIZE)

                # 原点の位置はマーカー中心の深度から求める
                mcx, mcy = img_pts.mean(axis=0)
                marker_org = get_3d_pt(mcx, mcy)

        if marker_R is None:
            cv2.putText(color_image, f"Marker ID {MARKER_ID}: not found", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # --- 2. ファントムの検出 ---
        results = model(color_image, verbose=False)
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
    cap.release()
    cv2.destroyAllWindows()
