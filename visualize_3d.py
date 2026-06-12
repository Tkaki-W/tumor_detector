import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 設定 ---
MODEL_PATH = r"C:\Users\mswas\Desktop\オクターブン卒論\Yolo操作\runs\detect\train-8\weights\best.pt"
model = YOLO(MODEL_PATH)

# RealSense初期化
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

align = rs.align(rs.stream.color)
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

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
            if d > 0:
                # 3D座標に変換
                pt = rs.rs2_deproject_pixel_to_point(intrinsics, [u + x_offset, v + y_offset], d)
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
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        results = model(color_image, verbose=False)
        
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
            rx2 = min(640, x2 + margin)
            ry2 = min(480, y2 + margin)
            
            # 深度ROIを取得
            depth_roi = depth_image[ry1:ry2, rx1:rx2]
            # 単位変換（16bit unsigned -> meters）
            depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            depth_roi_meters = depth_roi.astype(float) * depth_scale
            
            print("3Dモデル生成中...")
            show_3d_tumor(depth_roi_meters, intrinsics, rx1, ry1)
            
        elif key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
