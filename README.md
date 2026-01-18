# Stereo Matching 使用說明

本專案實作了基於 WCT (Weighted Census Transform) + Guided Image Filter + WTA 的立體匹配演算法。

## 安裝依賴

```bash
pip install numpy pillow
```

## 使用方法

### 方法一：命令列介面 (CLI)

使用 `stereo.py` 作為主程式，透過命令列參數執行：

```bash
python stereo.py --left <左影像路徑> --right <右影像路徑> --dmax <最大視差> [其他選項]
```

#### 必要參數

- `--left`: 左影像檔案路徑（支援常見影像格式：PNG, JPG, BMP 等）
- `--right`: 右影像檔案路徑
- `--dmax`: 最大視差數量（正整數，例如 64）

#### 可選參數

- `--wct_radius`: WCT 半徑（預設：4）
- `--base_weight`: WCT 基準權重（預設：8.0）
- `--guided_radius`: Guided Filter 視窗半徑（預設：3）
- `--guided_eps`: Guided Filter 正則化項（預設：1e-3）
- `--output`: 輸出視差圖路徑（PNG 格式，可選）
- `--output_npy`: 輸出視差 numpy 陣列路徑（.npy 格式，可選）
- `--progress`: 顯示簡單進度（可選）

#### 範例

```bash
# 基本使用
python stereo.py --left left.png --right right.png --dmax 64 --output disparity.png

# 完整參數範例
python stereo.py \
  --left left.png \
  --right right.png \
  --dmax 64 \
  --wct_radius 4 \
  --base_weight 8.0 \
  --guided_radius 3 \
  --guided_eps 0.001 \
  --output disparity.png \
  --output_npy disparity.npy \
  --progress
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
left_gray = to_gray(left_img)
right_gray = to_gray(right_img)

# 計算視差
disparity, min_cost = compute_disparity(
    left_gray=left_gray,
    right_gray=right_gray,
    dmax=64,
    wct_radius=4,
    base_weight=8.0,
    guided_radius=3,
    guided_eps=1e-3,
)

# disparity: 視差圖，形狀為 (H, W)，dtype 為 int32
# min_cost: 最小 cost 圖，形狀為 (H, W)，dtype 為 float32
```

#### 分步驟使用

```python
from stereo_io import read_image, to_gray
from census import compute_wct_cost_volume
from stereo import aggregate_cost_volume, winner_take_all

# 讀取與轉灰階
left_img = read_image("left.png")
right_img = read_image("right.png")
left_gray = to_gray(left_img)
right_gray = to_gray(right_img)

# 步驟 1: 計算 WCT cost volume
dsi = compute_wct_cost_volume(
    left=left_gray,
    right=right_gray,
    dmax=64,
    radius=4,
    base_weight=8.0,
)

# 步驟 2: Guided Filter 聚合
aggregated = aggregate_cost_volume(
    dsi=dsi,
    guide=left_gray,
    radius=3,
    eps=1e-3,
)

# 步驟 3: WTA 輸出視差
disparity, min_cost = winner_take_all(aggregated)
```

#### 單獨使用各模組

**影像 I/O (`stereo_io.py`)**

```python
from stereo_io import read_image, to_gray, ensure_same_shape

# 讀取影像
img = read_image("image.png")  # 回傳 numpy 陣列

# 轉灰階並正規化到 0~1
gray = to_gray(img)  # 回傳 float32，範圍 0~1

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
filtered = guided_filter(guide=left_gray, src=cost_layer, radius=3, eps=1e-3)
```

## 模組說明

### `stereo_io.py`
- `read_image(path)`: 讀取影像檔案
- `to_gray(image)`: 轉灰階並正規化到 0~1
- `ensure_same_shape(left, right)`: 確認兩影像尺寸一致

### `census.py`
- `generate_offsets(radius)`: 產生 8 方向位移清單
- `compute_weights(offsets, base_weight)`: 計算權重陣列
- `compute_wct_cost_volume(left, right, dmax, radius, base_weight, progress_callback=None)`: 計算 WCT cost volume

### `guided_filter.py`
- `integral_image(image)`: 計算 integral image
- `box_filter_mean(image, radius)`: Box filter 區域平均
- `guided_filter(guide, src, radius, eps)`: Guided Image Filter

### `stereo.py`
- `aggregate_cost_volume(dsi, guide, radius, eps, progress_callback=None)`: 對 cost volume 做聚合
- `winner_take_all(cost_volume)`: WTA 輸出視差
- `compute_disparity(...)`: 完整流程函式

## 參數建議

- **dmax**: 依影像內容與相機基線設定，常見值為 32、64、128
- **wct_radius**: 建議 3~5，影響 Census 特徵範圍
- **base_weight**: 預設 8.0，通常不需調整
- **guided_radius**: 建議 3~5，影響平滑程度
- **guided_eps**: 建議 1e-3 到 1e-1，越小越銳利但可能過度敏感

## 輸出說明

- **視差圖 (disparity)**: 每個像素的視差值，範圍為 0 到 dmax-1
- **最小 cost 圖 (min_cost)**: 可用於除錯，觀察哪些區域匹配困難

## 注意事項

1. 左右影像必須尺寸一致
2. 影像會自動轉為灰階並正規化到 0~1
3. 邊界處理：當 `(x-d) < 0` 或比較點超出範圍時，該 disparity 的 cost 會設為大值
4. 所有函式都有完整的型別標註與文件說明

## Middlebury 2014 stereo dataset description

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