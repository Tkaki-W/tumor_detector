#評価用と学習用の写真を分ける
#ラベル変更の際はyamlファイルを編集
import os
import shutil
import random

src_dir = r"C:\Users\mswas\Desktop\オクターブン卒論\Yolo操作\dataset\anotations\ファントム\obj_train_data"
dst_dir = r"C:\Users\mswas\Desktop\オクターブン卒論\Yolo操作\dataset"

# フォルダ作成
for split in ["train", "val"]:
    os.makedirs(f"{dst_dir}/images/{split}", exist_ok=True)
    os.makedirs(f"{dst_dir}/labels/{split}", exist_ok=True)

# 画像一覧取得
images = [f for f in os.listdir(src_dir) if f.endswith(".jpg")]
random.shuffle(images)

# 8:2でtrain/val分割
split_idx = int(len(images) * 0.8)
train_images = images[:split_idx]
val_images = images[split_idx:]

for split, imgs in [("train", train_images), ("val", val_images)]:
    for img in imgs:
        name = os.path.splitext(img)[0]
        shutil.copy(f"{src_dir}/{img}", f"{dst_dir}/images/{split}/{img}")
        txt = f"{name}.txt"
        if os.path.exists(f"{src_dir}/{txt}"):
            shutil.copy(f"{src_dir}/{txt}", f"{dst_dir}/labels/{split}/{txt}")

print("完了！")