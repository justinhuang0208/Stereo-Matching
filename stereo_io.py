from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from PIL import Image


def read_image(path: str) -> np.ndarray:
    """讀取影像並回傳為 numpy 陣列（保持原色彩通道）。

    參數:
        path: 影像檔案路徑。

    回傳:
        影像陣列，dtype 依原始影像而定。
    """
    image: Image.Image = Image.open(path)
    return np.array(image)


def _resolve_normalization_scale(image: np.ndarray, max_value: float | None) -> float:
    """取得正規化到 0~1 時的縮放上限值。

    參數:
        image: 輸入影像陣列。
        max_value: 指定的最大值，若為 None 則自動推斷。

    回傳:
        正規化用的最大值。
    """
    if max_value is not None:
        if max_value <= 0:
            raise ValueError("max_value 必須為正值。")
        return float(max_value)
    if np.issubdtype(image.dtype, np.integer):
        return float(np.iinfo(image.dtype).max)
    image_min: float = float(np.nanmin(image))
    image_max: float = float(np.nanmax(image))
    if image_min >= 0.0 and image_max <= 1.0:
        return 1.0
    if image_max <= 0.0:
        raise ValueError("影像最大值必須為正值，才能正規化到 0~1。")
    return image_max


def to_gray(
    image: np.ndarray,
    normalize: bool = False,
    max_value: float | None = None,
) -> np.ndarray:
    """將影像轉為灰階 float32，可選擇是否正規化到 0~1。

    參數:
        image: 輸入影像陣列，形狀為 HxW 或 HxWx3/4。
        normalize: 是否將輸出正規化到 0~1。
        max_value: 正規化時使用的最大值，None 代表自動推斷。

    回傳:
        灰階影像陣列，dtype 為 float32。
    """
    if image.ndim == 2:
        gray: np.ndarray = image.astype(np.float32)
    elif image.ndim == 3 and image.shape[2] >= 3:
        rgb = image[..., :3].astype(np.float32)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    else:
        raise ValueError("不支援的影像形狀")

    gray_f: np.ndarray = gray.astype(np.float32)
    if not normalize:
        return gray_f
    scale: float = _resolve_normalization_scale(image, max_value)
    normalized: np.ndarray = gray_f / np.float32(scale)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)



def ensure_same_shape(left: np.ndarray, right: np.ndarray) -> Tuple[int, int]:
    """確認左右影像尺寸一致，回傳高度與寬度。

    參數:
        left: 左影像灰階陣列。
        right: 右影像灰階陣列。

    回傳:
        (height, width)。
    """
    if left.shape != right.shape:
        raise ValueError("左右影像尺寸不一致。")
    if left.ndim != 2:
        raise ValueError("灰階影像維度必須為 2。")
    height: int = int(left.shape[0])
    width: int = int(left.shape[1])
    return height, width


def read_pfm(path: str) -> np.ndarray:
    """讀取 PFM 檔案並回傳 float32 陣列。

    參數:
        path: PFM 檔案路徑。

    回傳:
        影像陣列，灰階為 (H, W)，彩色為 (H, W, 3)。
    """
    with open(path, "rb") as handler:
        header: str = handler.readline().decode("ascii").strip()
        if header not in ("PF", "Pf"):
            raise ValueError("PFM header 必須為 PF 或 Pf。")
        color: bool = header == "PF"

        def _read_non_empty_line() -> str:
            line: str = handler.readline().decode("ascii")
            while line:
                stripped: str = line.strip()
                if stripped and not stripped.startswith("#"):
                    return stripped
                line = handler.readline().decode("ascii")
            raise ValueError("PFM 內容不完整。")

        dim_line: str = _read_non_empty_line()
        width_str, height_str = dim_line.split()
        width: int = int(width_str)
        height: int = int(height_str)
        scale_line: str = _read_non_empty_line()
        scale: float = float(scale_line)
        endian: str = "<" if scale < 0 else ">"
        channels: int = 3 if color else 1
        count: int = width * height * channels
        data: np.ndarray = np.fromfile(handler, dtype=f"{endian}f", count=count)
        if data.size != count:
            raise ValueError("PFM 資料大小不正確。")
        if color:
            image: np.ndarray = data.reshape((height, width, 3))
        else:
            image = data.reshape((height, width))
        image = np.flipud(image).astype(np.float32)
        return image


def read_pgm_mask(path: str) -> np.ndarray:
    """讀取 PGM 遮罩並回傳布林陣列。

    參數:
        path: PGM 遮罩檔案路徑。

    回傳:
        遮罩陣列，True 表示有效像素。
    """
    image: np.ndarray = read_image(path)
    if image.ndim == 3:
        image = image[..., 0]
    if image.ndim != 2:
        raise ValueError("PGM 遮罩必須為 2D。")
    return image > 0


def write_pfm(path: str, image: np.ndarray, scale: float = 1.0) -> None:
    """將 float32 陣列寫入 PFM 檔案。

    參數:
        path: 輸出 PFM 檔案路徑。
        image: 影像陣列，形狀為 (H, W) 或 (H, W, 3)。
        scale: 實數縮放值，正負號決定 endian。

    回傳:
        None。
    """
    if image.ndim not in (2, 3):
        raise ValueError("PFM 影像維度必須為 2 或 3。")
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError("彩色 PFM 必須為 3 通道。")

    color: bool = image.ndim == 3
    height: int = int(image.shape[0])
    width: int = int(image.shape[1])
    header: str = "PF" if color else "Pf"
    data: np.ndarray = np.flipud(image).astype(np.float32)
    endian: str = "<" if data.dtype.byteorder in ("<", "=") else ">"
    scale_value: float = -abs(scale) if endian == "<" else abs(scale)

    with open(path, "wb") as handler:
        handler.write(f"{header}\n".encode("ascii"))
        handler.write(f"{width} {height}\n".encode("ascii"))
        handler.write(f"{scale_value}\n".encode("ascii"))
        data.tofile(handler)


def save_npz(path: str, arrays: Dict[str, np.ndarray]) -> None:
    """將多個陣列儲存為 NPZ 檔案。

    參數:
        path: 輸出 NPZ 檔案路徑。
        arrays: 欲輸出的陣列字典。

    回傳:
        None。
    """
    if not arrays:
        raise ValueError("arrays 不可為空。")
    np.savez_compressed(path, **arrays)


def load_npz(path: str) -> Dict[str, np.ndarray]:
    """讀取 NPZ 檔案並回傳陣列字典。

    參數:
        path: NPZ 檔案路徑。

    回傳:
        陣列字典。
    """
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def save_disparity_npz(path: str, disparity: np.ndarray, min_cost: np.ndarray) -> None:
    """儲存 disparity 與 min_cost 至 NPZ。

    參數:
        path: 輸出 NPZ 檔案路徑。
        disparity: 視差圖陣列。
        min_cost: 最小成本圖陣列。

    回傳:
        None。
    """
    if disparity.shape != min_cost.shape:
        raise ValueError("disparity 與 min_cost 尺寸必須一致。")
    save_npz(path, {"disparity": disparity, "min_cost": min_cost})


def load_disparity_npz(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """讀取 disparity 與 min_cost 的 NPZ 檔案。

    參數:
        path: NPZ 檔案路徑。

    回傳:
        (disparity, min_cost)。
    """
    data: Dict[str, np.ndarray] = load_npz(path)
    if "disparity" not in data or "min_cost" not in data:
        raise ValueError("NPZ 必須包含 disparity 與 min_cost。")
    return data["disparity"], data["min_cost"]


def convert_npz_to_pfm(npz_path: str, pfm_path: str, key: str = "disparity") -> None:
    """將 NPZ 內指定 key 的陣列轉為 PFM。

    參數:
        npz_path: NPZ 檔案路徑。
        pfm_path: 輸出 PFM 檔案路徑。
        key: NPZ 內的陣列 key。

    回傳:
        None。
    """
    data: Dict[str, np.ndarray] = load_npz(npz_path)
    if key not in data:
        raise ValueError(f"NPZ 不包含 key: {key}")
    write_pfm(pfm_path, data[key])


def convert_pfm_to_npz(pfm_path: str, npz_path: str, key: str = "disparity") -> None:
    """將 PFM 轉為 NPZ，指定 key 儲存。

    參數:
        pfm_path: PFM 檔案路徑。
        npz_path: 輸出 NPZ 檔案路徑。
        key: NPZ 內的陣列 key。

    回傳:
        None。
    """
    data: np.ndarray = read_pfm(pfm_path)
    save_npz(npz_path, {key: data})
