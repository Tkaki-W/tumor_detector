# Tumor Detector

**YOLOv8 × 深度カメラによる腫瘍ファントムの検出と Z 座標測距**

自動触診ロボットのアーム制御に向けて、深度カメラの映像から YOLO で腫瘍ファントムを検出し、
カメラから鉛直下向きの距離（Z 座標）や AR マーカー基準の位置をリアルタイムに測定します。

深度カメラごとに同じ構成のフォルダを用意しています。処理の流れや測距の方法は共通で、カメラまわりの部分だけが違います。

| フォルダ | カメラ | 詳細 |
| --- | --- | --- |
| [`realsense/`](realsense/) | Intel RealSense（640×480） | [README](realsense/README.md) |
| [`zed2/`](zed2/) | Stereolabs ZED2（1280×720） | [README](zed2/README.md) |

## 2 つの版の違い

| | RealSense | ZED2 |
| --- | --- | --- |
| ライブラリ | `pyrealsense2` | `pyzed`（ZED SDK 5.5 の `get_python_api.py` で導入） |
| 解像度 | 640×480 | 1280×720（HD720） |
| 深度と画像の位置合わせ | `rs.align` | 不要（左カメラ画像に合わせて出力される） |
| YOLO の推論サイズ | 640 | 1280（広角でファントムが小さく写るため） |

## 各フォルダの中身

| ファイル | 内容 |
| --- | --- |
| `capture.py` | 学習用画像の撮影 |
| `prepare_dataset.py` | アノテーション済みデータを train/val に 8:2 で分割 |
| `Yolo_train.py` | 学習 |
| `eval.py` | 検出 + Z 測距 |
| `aruco_eval.py` | AR マーカー（ID 0）を原点としたファントムの位置 |
| `visualize_3d.py` | 検出したファントムの表面を 3D 表示 |

スクリプト内のパスはそれぞれのフォルダからの相対パスなので、どこから実行しても動きます。
