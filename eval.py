# ptファイルを呼び出して対象をくくる
import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO

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

# 深度フレームをカラーフレームに合わせる
align_to = rs.stream.color
align = rs.align(align_to)

# ウィンドウの設定：WINDOW_NORMALにすることで、マウスで拡大・縮小が可能になります
window_name = "Tumor Detection (Resizable)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

print("腫瘍半径検知モード: 開始 ('q'キーで終了)")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        results = model(color_image, verbose=False)
        
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
                    radius_mm = (radius_px * dist_center * 1000 / fx) * 1.15
                else:
                    radius_mm = 0
                
                # ラベル作成
                cls = int(box.cls[0])
                cls_name = model.names[cls]
                conf = float(box.conf[0])
                line1 = f"{cls_name} ({conf:.2f})"
                line2 = f"R: {radius_mm:.1f}mm CD: {dist_center:.2f}m"
                
                # 描画
                color = (0, 255, 0)
                cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(color_image, line1, (x1, y1 - 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(color_image, line2, (x1, y1 - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 中心点の可視化
                cv2.circle(color_image, (cx, cy), 4, (255, 0, 0), -1)

        cv2.imshow(window_name, color_image)
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
