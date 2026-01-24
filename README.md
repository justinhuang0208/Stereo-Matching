# Stereo Matching 使用說明

本專案實作了基於 WCT (Weighted Census Transform) + 可選擇濾波器 + WTA 的立體匹配演算法。

## 安裝依賴

```bash
pip install numpy pillow opencv-python
```

## 使用方法

### 方法一：命令列介面 (CLI)

使用 `stereo.py` 作為主程式，透過命令列參數執行：

```bash
python stereo.py --left <左影像路徑> --right <右影像路徑> --dmax <最大視差> --gt <GT PFM 路徑> --gt-mask <GT 遮罩 PGM 路徑> [其他選項]
```

#### 必要參數（未使用 dataset 模式）

- `--left`: 左影像檔案路徑（支援常見影像格式：PNG, JPG, BMP 等）
- `--right`: 右影像檔案路徑
- `--dmax`: 最大視差數量（正整數，例如 64）
- `--gt`: GT PFM 檔案路徑（預設所有執行皆會評估）
- `--gt-mask`: GT 遮罩 PGM 檔案路徑（disp0-n.pgm）

#### 可選參數

- `--dataset`: dataset 資料夾名稱（自動使用 `im0.png`、`im1.png`、`disp0.pfm`、`disp0-n.pgm`，並從 `ndisp_summary.csv` 取得 `dmax`）
- `--all-datasets`: 批次處理 `dataset/` 下所有場景並輸出彙總評估
- `--wct_radius`: WCT 半徑（預設：4）
- `--base_weight`: WCT 基準權重（預設：8.0）
- `--guided_radius`: Guided Filter 視窗半徑（預設：3）
- `--guided_eps`: Guided Filter 正則化項（預設：0.0154，依影像尺度調整）
- `--filter`: 聚合濾波器類型（guided, median, gaussian, bilateral）
- `--median_radius`: Median Filter 視窗半徑（預設：3，預設使用 OpenCV medianBlur）
- `--gaussian_sigma`: Gaussian sigma（預設：1.0）
- `--bilateral_sigma`: Bilateral sigma（預設：1.0）
- `--bad_threshold`: Bad pixel 閾值（預設：1.0）
- 預設會顯示簡單進度

#### 範例

```bash
# 基本使用
python stereo.py --left left.png --right right.png --dmax 64

# 完整參數範例
python stereo.py \
  --left left.png \
  --right right.png \
  --dmax 64 \
  --wct_radius 4 \
  --base_weight 8.0 \
  --guided_radius 3 \
  --guided_eps 0.0154 \
  --filter guided \
  --median_radius 3 \
  --gaussian_sigma 1.0 \
  --bilateral_sigma 1.0

# 含 GT 評估的範例（預設皆會評估）
python stereo.py \
  --left im0.png \
  --right im1.png \
  --dmax 64 \
  --gt dataset/Motorcycle-perfect/disp0.pfm \
  --gt-mask dataset/Motorcycle-perfect/disp0-n.pgm \
  --bad_threshold 1.0

# 使用 dataset 名稱（自動對應 im0/im1/disp0 並取得 dmax）
python stereo.py --dataset Motorcycle-perfect

# 批次處理 dataset 內所有場景（輸出 metrics_summary.json）
python stereo.py --all-datasets
```

### 方法一補充：使用既有 NPZ 重新評估

如果你已有輸出的 `disparity.npz`，可透過 `eval_npz.py` 重新指定 `bad_threshold` 進行評估：

```bash
# 基本使用
python eval_npz.py --npz result/202601200157/disparity.npz --gt dataset/Motorcycle-perfect/disp0.pfm --gt_mask dataset/Motorcycle-perfect/disp0-n.pgm --bad_threshold 2.0

# 當 NPZ 的 key 不是 disparity 時
python eval_npz.py --npz result/202601200157/disparity.npz --gt dataset/Motorcycle-perfect/disp0.pfm --gt_mask dataset/Motorcycle-perfect/disp0-n.pgm --key my_disp --bad_threshold 1.0

# 輸出評估結果到 JSON
python eval_npz.py --npz result/202601200157/disparity.npz --gt dataset/Motorcycle-perfect/disp0.pfm --gt_mask dataset/Motorcycle-perfect/disp0-n.pgm --bad_threshold 1.0 --output_json result/202601200157/metrics_custom.json
```

### 方法二：作為 Python 模組使用

#### 完整流程

```python
from stereo_io import read_image, to_gray
from stereo import compute_disparity

# 讀取影像
left_img = read_image("left.png")
right_img = read_image("right.png")

# 轉灰階
left_gray = to_gray(left_img, normalize=True)
right_gray = to_gray(right_img, normalize=True)

# 計算視差
disparity, min_cost = compute_disparity(
    left_gray=left_gray,
    right_gray=right_gray,
    dmax=64,
    wct_radius=4,
    base_weight=8.0,
    guided_radius=3,
    guided_eps=0.01,
    filter_type="guided",
    median_radius=3,
    gaussian_sigma=1.0,
    bilateral_sigma=1.0,
)

# disparity: 視差圖，形狀為 (H, W)，dtype 為 int32
# min_cost: 最小 cost 圖，形狀為 (H, W)，dtype 為 float32
```

#### 分步驟使用

```python
from stereo_io import read_image, to_gray
from census import compute_wct_cost_volume
from stereo import aggregate_and_wta

# 讀取與轉灰階
left_img = read_image("left.png")
right_img = read_image("right.png")
left_gray = to_gray(left_img, normalize=True)
right_gray = to_gray(right_img, normalize=True)

# 步驟 1: 計算 WCT cost volume
dsi = compute_wct_cost_volume(
    left=left_gray,
    right=right_gray,
    dmax=64,
    radius=4,
    base_weight=8.0,
)

# 步驟 2: 濾波聚合並 WTA
disparity, min_cost = aggregate_and_wta(
    dsi=dsi,
    guide=left_gray,
    guided_radius=3,
    guided_eps=0.01,
    filter_type="guided",
    median_radius=3,
    gaussian_sigma=1.0,
    bilateral_sigma=1.0,
)
```

#### 單獨使用各模組

**影像 I/O (`stereo_io.py`)**

```python
from stereo_io import read_image, to_gray, ensure_same_shape

# 讀取影像
img = read_image("image.png")  # 回傳 numpy 陣列

# 轉灰階（可正規化到 0~1）
gray = to_gray(img, normalize=True)  # 回傳 float32

# 確認兩影像尺寸一致
h, w = ensure_same_shape(left_gray, right_gray)
```

**Census Transform (`census.py`)**

```python
from census import generate_offsets, compute_weights, compute_wct_cost_volume

# 產生位移清單（8 方向，距離 1..radius）
offsets = generate_offsets(radius=4)  # 32 個位移

# 計算權重
weights = compute_weights(offsets, base_weight=8.0)

# 計算 cost volume
dsi = compute_wct_cost_volume(left, right, dmax=64, radius=4, base_weight=8.0)
```

**Guided Filter (`guided_filter.py`)**

```python
from guided_filter import integral_image, box_filter_mean, guided_filter

# 計算 integral image
integral = integral_image(image)

# Box filter 區域平均
mean = box_filter_mean(image, radius=3)

# Guided Filter
filtered = guided_filter(guide=left_gray, src=cost_layer, radius=3, eps=0.01)
```

**Filtering (`filters.py`)**

```python
from filters import median_filter, gaussian_filter, bilateral_filter

median = median_filter(cost_layer, radius=3)
gaussian = gaussian_filter(cost_layer, sigma=1.0, method="opencv")
bilateral = bilateral_filter(cost_layer, sigma=1.0)
```

## 模組說明

### `stereo_io.py`
- `read_image(path)`: 讀取影像檔案
- `to_gray(image, normalize=False, max_value=None)`: 轉灰階，可選擇正規化到 0~1
- `ensure_same_shape(left, right)`: 確認兩影像尺寸一致

### `census.py`
- `generate_offsets(radius)`: 產生 8 方向位移清單
- `compute_weights(offsets, base_weight)`: 計算權重陣列
- `compute_wct_cost_volume(left, right, dmax, radius, base_weight, progress_callback=None)`: 計算 WCT cost volume

### `guided_filter.py`
- `integral_image(image)`: 計算 integral image
- `box_filter_mean(image, radius)`: Box filter 區域平均
- `guided_filter(guide, src, radius, eps)`: Guided Image Filter

### `filters.py`
- `median_filter(image, radius)`: Median Filter
- `gaussian_filter(image, sigma, method="opencv")`: Gaussian Filter
- `bilateral_filter(image, sigma)`: Bilateral Filter

### `stereo.py`
- `aggregate_and_wta(...)`: 逐層聚合並即時 WTA 輸出視差
- `compute_disparity(...)`: 完整流程函式

## 參數建議

- **dmax**: 依影像內容與相機基線設定，常見值為 32、64、128
- **wct_radius**: 建議 3~5，影響 Census 特徵範圍
- **base_weight**: 預設 8.0，通常不需調整
- **guided_radius**: 建議 3~5，影響平滑程度
- **guided_eps**: 影像正規化到 0~1 時，常見範圍為 0.001~0.01

## 輸出說明

- **輸出資料夾**: 每次執行會輸出到 `result/<YYYYMMDDHHMMSS>`，同秒重複執行會自動加上 `_01`、`_02` 避免覆寫，且最後六碼維持為 `HHMMSS`
- **視差圖 (disparity.png)**: 每個像素的視差值，範圍為 0 到 dmax-1
- **彩色視差圖 (disparity_color.png)**: 以 Jet 色盤呈現視差分佈，便於視覺化
- **原始資料 (disparity.npz)**: `disparity` 與 `min_cost` 的原始陣列
- **參數檔 (params.json)**: 這次執行的所有參數與輸出路徑
- **評估結果 (metrics.json)**: 含 PBM 與 RMS 的評估結果（僅在啟用 `--eval` 時）

## 注意事項

1. 左右影像必須尺寸一致
2. 影像會自動轉為灰階，保留原始亮度範圍
3. 邊界處理：當 `(x-d) < 0` 或比較點超出範圍時，該 disparity 的 cost 會設為大值
4. 所有函式都有完整的型別標註與文件說明

## 原始資料與 PFM 轉換

可使用 `convert.py` 進行 `.npz` 與 `.pfm` 的互轉：

```bash
# NPZ -> PFM
python convert.py --input result/202601191214/disparity.npz --output disparity.pfm --mode npz2pfm

# PFM -> NPZ
python convert.py --input dataset/Motorcycle-perfect/disp0.pfm --output disp0.npz --mode pfm2npz
```

## Middlebury 2014 stereo dataset description

```
Each dataset consists of 2 views taken under several different illuminations and exposures. The files are organized as follows:
SCENE-{perfect,imperfect}/     -- each scene comes with perfect and imperfect calibration (see paper)
  ambient/                     -- directory of all input views under ambient lighting
    L{1,2,...}/                -- different lighting conditions
      im0e{0,1,2,...}.png      -- left view under different exposures
      im1e{0,1,2,...}.png      -- right view under different exposures
  calib.txt                    -- calibration information
  im{0,1}.png                  -- default left and right view
  im1E.png                     -- default right view under different exposure
  im1L.png                     -- default right view with different lighting
  disp{0,1}.pfm                -- left and right GT disparities
  disp{0,1}-n.png              -- left and right GT number of samples (* perfect only)
  disp{0,1}-sd.pfm             -- left and right GT sample standard deviations (* perfect only)
  disp{0,1}y.pfm               -- left and right GT y-disparities (* imperfect only)
```