from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import numba


_NUMBA_NJIT: Callable[[Callable[..., object]], Callable[..., object]] = numba.njit(cache=True, fastmath=False)


def _should_use_numba(use_numba: bool) -> bool:
    """判斷是否使用 Numba JIT。"""
    return bool(use_numba)


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


@_NUMBA_NJIT
def _integral_image_numba(image: np.ndarray) -> np.ndarray:
    """以 Numba 計算 integral image，回傳大小為 (H+1, W+1)。"""
    height: int = int(image.shape[0])
    width: int = int(image.shape[1])
    integral: np.ndarray = np.zeros((height + 1, width + 1), dtype=np.float32)
    temp: np.ndarray = np.empty((height, width), dtype=np.float32)
    for x in range(width):
        col_sum: np.float32 = np.float32(0.0)
        for y in range(height):
            col_sum = np.float32(col_sum + image[y, x])
            temp[y, x] = col_sum
    for y in range(height):
        row_sum = np.float32(0.0)
        for x in range(width):
            row_sum = np.float32(row_sum + temp[y, x])
            integral[y + 1, x + 1] = row_sum
    return integral


@dataclass(frozen=True)
class BoxFilterIndex:
    """Box filter 所需的索引與區域面積。"""

    y0: np.ndarray
    y1: np.ndarray
    x0: np.ndarray
    x1: np.ndarray
    area: np.ndarray


def _prepare_box_filter_index(height: int, width: int, radius: int) -> BoxFilterIndex:
    """預先計算 box filter 的索引與區域面積。"""
    if height <= 0 or width <= 0:
        raise ValueError("height 與 width 必須為正整數。")
    if radius < 0:
        raise ValueError("radius 必須為非負整數。")
    ys: np.ndarray = np.arange(height)
    xs: np.ndarray = np.arange(width)
    y0: np.ndarray = np.clip(ys - radius, 0, height - 1)
    y1: np.ndarray = np.clip(ys + radius, 0, height - 1)
    x0: np.ndarray = np.clip(xs - radius, 0, width - 1)
    x1: np.ndarray = np.clip(xs + radius, 0, width - 1)
    area: np.ndarray = (y1 - y0 + 1)[:, None] * (x1 - x0 + 1)[None, :]
    return BoxFilterIndex(y0=y0, y1=y1, x0=x0, x1=x1, area=area.astype(np.float32))


@dataclass(frozen=True)
class GuidedFilterPrecomputed:
    """Guided Filter 的預先計算結果。"""

    guide_f: np.ndarray
    indices: BoxFilterIndex
    mean_guide: np.ndarray
    mean_gg: np.ndarray
    var_g: np.ndarray
    radius: int
    eps: float
    use_numba: bool


def _box_sum_from_integral(
    integral: np.ndarray,
    indices: BoxFilterIndex,
) -> np.ndarray:
    """使用 integral image 計算每個像素的視窗總和。

    參數:
        integral: integral image，大小為 (H+1, W+1)。
        indices: box filter 的索引與區域面積。

    回傳:
        每個像素的視窗總和，形狀為 (H, W)。
    """
    sum_region: np.ndarray = (
        integral[indices.y1[:, None] + 1, indices.x1[None, :] + 1]
        - integral[indices.y0[:, None], indices.x1[None, :] + 1]
        - integral[indices.y1[:, None] + 1, indices.x0[None, :]]
        + integral[indices.y0[:, None], indices.x0[None, :]]
    )
    return sum_region.astype(np.float32)


@_NUMBA_NJIT
def _box_sum_from_integral_numba(
    integral: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
) -> np.ndarray:
    """以 Numba 計算 box sum，回傳形狀為 (H, W)。"""
    height: int = int(y0.shape[0])
    width: int = int(x0.shape[0])
    sum_region: np.ndarray = np.empty((height, width), dtype=np.float32)
    for y in range(height):
        y0v: int = int(y0[y])
        y1v: int = int(y1[y])
        for x in range(width):
            x0v: int = int(x0[x])
            x1v: int = int(x1[x])
            sum_region[y, x] = (
                integral[y1v + 1, x1v + 1]
                - integral[y0v, x1v + 1]
                - integral[y1v + 1, x0v]
                + integral[y0v, x0v]
            )
    return sum_region


def box_filter_mean(image: np.ndarray, radius: int, use_numba: bool = True) -> np.ndarray:
    """計算 box filter 的區域平均。

    參數:
        image: 輸入 2D 影像。
        radius: 視窗半徑。
        use_numba: 是否啟用 Numba JIT。

    回傳:
        區域平均影像，形狀為 (H, W)。
    """
    indices: BoxFilterIndex = _prepare_box_filter_index(image.shape[0], image.shape[1], radius)
    return box_filter_mean_with_indices(image, indices, use_numba=use_numba)


def box_filter_mean_with_indices(
    image: np.ndarray,
    indices: BoxFilterIndex,
    use_numba: bool = True,
) -> np.ndarray:
    """使用預先計算的索引來計算 box filter 的區域平均。"""
    if image.ndim != 2:
        raise ValueError("image 必須為 2D。")
    image_f: np.ndarray = image.astype(np.float32, copy=False)
    if _should_use_numba(use_numba):
        integral = _integral_image_numba(image_f)
        sum_region = _box_sum_from_integral_numba(
            integral,
            indices.y0,
            indices.y1,
            indices.x0,
            indices.x1,
        )
    else:
        integral = integral_image(image_f)
        sum_region = _box_sum_from_integral(integral, indices)
    return sum_region / indices.area


def prepare_guided_filter(
    guide: np.ndarray,
    radius: int,
    eps: float,
    use_numba: bool = True,
) -> GuidedFilterPrecomputed:
    """預先計算 Guided Filter 所需的 guide 統計量。

    參數:
        guide: 引導影像（灰階）。
        radius: 視窗半徑。
        eps: 正則化項。
        use_numba: 是否啟用 Numba JIT。

    回傳:
        GuidedFilterPrecomputed，包含預先計算的統計量。
    """
    if guide.ndim != 2:
        raise ValueError("guide 必須為 2D。")
    if radius <= 0:
        raise ValueError("radius 必須為正整數。")
    if eps <= 0:
        raise ValueError("eps 必須為正值。")
    guide_f: np.ndarray = guide.astype(np.float32)
    indices: BoxFilterIndex = _prepare_box_filter_index(guide_f.shape[0], guide_f.shape[1], radius)
    mean_guide: np.ndarray = box_filter_mean_with_indices(guide_f, indices, use_numba=use_numba)
    mean_gg: np.ndarray = box_filter_mean_with_indices(guide_f * guide_f, indices, use_numba=use_numba)
    var_g: np.ndarray = mean_gg - mean_guide * mean_guide
    return GuidedFilterPrecomputed(
        guide_f=guide_f,
        indices=indices,
        mean_guide=mean_guide,
        mean_gg=mean_gg,
        var_g=var_g,
        radius=radius,
        eps=float(eps),
        use_numba=use_numba,
    )


def guided_filter_with_precompute(
    precomputed: GuidedFilterPrecomputed,
    src: np.ndarray,
) -> np.ndarray:
    """使用預先計算的 guide 統計量執行 Guided Filter。

    參數:
        precomputed: GuidedFilterPrecomputed。
        src: 輸入影像（要被濾波的 cost）。

    回傳:
        濾波後影像，形狀為 (H, W)。
    """
    if src.ndim != 2:
        raise ValueError("src 必須為 2D。")
    if src.shape != precomputed.guide_f.shape:
        raise ValueError("src 與 guide 尺寸必須一致。")
    src_f: np.ndarray = src.astype(np.float32)
    mean_src: np.ndarray = box_filter_mean_with_indices(src_f, precomputed.indices, use_numba=precomputed.use_numba)
    mean_gs: np.ndarray = box_filter_mean_with_indices(
        precomputed.guide_f * src_f,
        precomputed.indices,
        use_numba=precomputed.use_numba,
    )
    cov_gs: np.ndarray = mean_gs - precomputed.mean_guide * mean_src
    a: np.ndarray = cov_gs / (precomputed.var_g + np.float32(precomputed.eps))
    b: np.ndarray = mean_src - a * precomputed.mean_guide
    mean_a: np.ndarray = box_filter_mean_with_indices(a, precomputed.indices, use_numba=precomputed.use_numba)
    mean_b: np.ndarray = box_filter_mean_with_indices(b, precomputed.indices, use_numba=precomputed.use_numba)
    q: np.ndarray = mean_a * precomputed.guide_f + mean_b
    return q.astype(np.float32)


def guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int,
    eps: float,
    use_numba: bool = True,
) -> np.ndarray:
    """使用 Guided Image Filter 對 src 做平滑。

    參數:
        guide: 引導影像（灰階）。
        src: 輸入影像（要被濾波的 cost）。
        radius: 視窗半徑。
        eps: 正則化項。
        use_numba: 是否啟用 Numba JIT。

    回傳:
        濾波後影像，形狀為 (H, W)。
    """
    if guide.shape != src.shape:
        raise ValueError("guide 與 src 尺寸必須一致。")
    precomputed: GuidedFilterPrecomputed = prepare_guided_filter(guide, radius, eps, use_numba=use_numba)
    return guided_filter_with_precompute(precomputed, src)
