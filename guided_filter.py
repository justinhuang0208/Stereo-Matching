from __future__ import annotations

from typing import Tuple

import numpy as np


def integral_image(image: np.ndarray) -> np.ndarray:
    """計算 integral image，回傳大小為 (H+1, W+1)。

    參數:
        image: 輸入 2D 影像。

    回傳:
        integral image，形狀為 (H+1, W+1)。
    """
    if image.ndim != 2:
        raise ValueError("image 必須為 2D。")
    integral: np.ndarray = np.zeros((image.shape[0] + 1, image.shape[1] + 1), dtype=np.float32)
    integral[1:, 1:] = np.cumsum(np.cumsum(image, axis=0), axis=1)
    return integral


def _box_sum_from_integral(integral: np.ndarray, radius: int) -> np.ndarray:
    """使用 integral image 計算每個像素的視窗總和。

    參數:
        integral: integral image，大小為 (H+1, W+1)。
        radius: 視窗半徑。

    回傳:
        每個像素的視窗總和，形狀為 (H, W)。
    """
    if radius < 0:
        raise ValueError("radius 必須為非負整數。")
    height: int = integral.shape[0] - 1
    width: int = integral.shape[1] - 1

    ys: np.ndarray = np.arange(height)
    xs: np.ndarray = np.arange(width)
    y0: np.ndarray = np.clip(ys - radius, 0, height - 1)
    y1: np.ndarray = np.clip(ys + radius, 0, height - 1)
    x0: np.ndarray = np.clip(xs - radius, 0, width - 1)
    x1: np.ndarray = np.clip(xs + radius, 0, width - 1)

    sum_region: np.ndarray = (
        integral[y1[:, None] + 1, x1[None, :] + 1]
        - integral[y0[:, None], x1[None, :] + 1]
        - integral[y1[:, None] + 1, x0[None, :]]
        + integral[y0[:, None], x0[None, :]]
    )
    return sum_region.astype(np.float32)


def box_filter_mean(image: np.ndarray, radius: int) -> np.ndarray:
    """計算 box filter 的區域平均。

    參數:
        image: 輸入 2D 影像。
        radius: 視窗半徑。

    回傳:
        區域平均影像，形狀為 (H, W)。
    """
    integral: np.ndarray = integral_image(image.astype(np.float32))
    sum_region: np.ndarray = _box_sum_from_integral(integral, radius)
    height: int = image.shape[0]
    width: int = image.shape[1]
    ys: np.ndarray = np.arange(height)
    xs: np.ndarray = np.arange(width)
    y0: np.ndarray = np.clip(ys - radius, 0, height - 1)
    y1: np.ndarray = np.clip(ys + radius, 0, height - 1)
    x0: np.ndarray = np.clip(xs - radius, 0, width - 1)
    x1: np.ndarray = np.clip(xs + radius, 0, width - 1)
    area: np.ndarray = (y1 - y0 + 1)[:, None] * (x1 - x0 + 1)[None, :]
    return sum_region / area.astype(np.float32)


def guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int,
    eps: float,
) -> np.ndarray:
    """使用 Guided Image Filter 對 src 做平滑。

    參數:
        guide: 引導影像（灰階）。
        src: 輸入影像（要被濾波的 cost）。
        radius: 視窗半徑。
        eps: 正則化項。

    回傳:
        濾波後影像，形狀為 (H, W)。
    """
    if guide.shape != src.shape:
        raise ValueError("guide 與 src 尺寸必須一致。")
    if guide.ndim != 2:
        raise ValueError("guide 與 src 必須為 2D。")
    if radius <= 0:
        raise ValueError("radius 必須為正整數。")
    if eps <= 0:
        raise ValueError("eps 必須為正值。")

    guide_f: np.ndarray = guide.astype(np.float32)
    src_f: np.ndarray = src.astype(np.float32)

    mean_guide: np.ndarray = box_filter_mean(guide_f, radius)
    mean_src: np.ndarray = box_filter_mean(src_f, radius)
    mean_gg: np.ndarray = box_filter_mean(guide_f * guide_f, radius)
    mean_gs: np.ndarray = box_filter_mean(guide_f * src_f, radius)

    var_g: np.ndarray = mean_gg - mean_guide * mean_guide
    cov_gs: np.ndarray = mean_gs - mean_guide * mean_src

    a: np.ndarray = cov_gs / (var_g + np.float32(eps))
    b: np.ndarray = mean_src - a * mean_guide

    mean_a: np.ndarray = box_filter_mean(a, radius)
    mean_b: np.ndarray = box_filter_mean(b, radius)

    q: np.ndarray = mean_a * guide_f + mean_b
    return q.astype(np.float32)
