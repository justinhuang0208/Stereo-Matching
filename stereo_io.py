from __future__ import annotations

from typing import Tuple

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


def to_gray(image: np.ndarray) -> np.ndarray:
    """將影像轉為灰階 float32，並正規化到 0~1。

    參數:
        image: 輸入影像陣列，形狀為 HxW 或 HxWx3/4。

    回傳:
        灰階影像陣列，dtype 為 float32，範圍 0~1。
    """
    if image.ndim == 2:
        gray: np.ndarray = image.astype(np.float32)
    elif image.ndim == 3 and image.shape[2] >= 3:
        rgb: np.ndarray = image[..., :3].astype(np.float32)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    else:
        raise ValueError("不支援的影像形狀，需為 HxW 或 HxWx3/4。")

    if gray.max() > 1.0:
        gray = gray / 255.0

    return gray.astype(np.float32)


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
