# ptファイルを呼び出して対象をくくる
import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO

# --- 設定項目 ---
# カメラの光軸（正面）と重力方向（鉛直下向き）のなす角度（度数法）逐一やってね
# カメラが真下を向いているなら 0、水平を向いているなら 90、斜め下45度なら 45
CAMERA_ANGLE_DEG = 49.0

# Yoloのrunsの中のptファイルを呼び出す
model = YOLO(r"C:\Users\mswas\Desktop\オクターブン卒論\Yolo操作\runs\detect\train-8\weights\best.pt")

pipeline = rs.pipeline()
config = rs.config()

# カラーと深度のストリームを有効化
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# ストリーミング開始
profile = pipeline.start(config)

# カメラの内部パラメータ（焦点距離など）を取得して実寸計算に使用
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
fx = intrinsics.fx
fy = intrinsics.fy
cam_cx = intrinsics.width // 2
cam_cy = intrinsics.height // 2

# 深度フレームをカラーフレームに合わせる
align_to = rs.stream.color
align = rs.align(align_to)

# ウィンドウの設定：WINDOW_NORMALにすることで、拡大縮小可能に
window_name = "Tumor Detection (Resizable)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

print("腫瘍半径検知モード: 開始 ('q'キーで終了)")

try:
    while True:
        frames = pipeline.wait_for_frames()
        #カラーと深度を別々に取り出す
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            continue
        
        color_image = np.asanyarray(color_frame.get_data())
        #機械学習モデルにかける！
        results = model(color_image, verbose=False)
        
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
                            if 0 <= nx < 640 and 0 <= ny < 480:
                                d = depth_frame.get_distance(nx, ny)
                                if d > 0: d_list.append(d)
                    if not d_list: return None
                    avg_d = np.mean(d_list)
                    # ピクセル座標から3D空間座標(x, y, z)に変換
                    return rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], avg_d), avg_d

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
                line4 = f"Z(vert): {vertical_dist:.2f}m"
                
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
    pipeline.stop()
    cv2.destroyAllWindows()
