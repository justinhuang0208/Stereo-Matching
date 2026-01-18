from __future__ import annotations

import importlib
import importlib.util
import math
from typing import Callable, Optional, Tuple

import numpy as np

_SCIPY_MEDIAN_FILTER: Optional[Callable[..., np.ndarray]] = None
_SCIPY_CHECKED: bool = False


def _validate_2d(image: np.ndarray, name: str) -> None:
    """檢查輸入影像為 2D。"""
    if image.ndim != 2:
        raise ValueError(f"{name} 必須為 2D。")


def _gaussian_kernel(radius: int, sigma: float) -> np.ndarray:
    """產生 2D Gaussian kernel。"""
    if radius <= 0:
        raise ValueError("radius 必須為正整數。")
    if sigma <= 0:
        raise ValueError("sigma 必須為正值。")
    size: int = radius * 2 + 1
    ax: np.ndarray = np.arange(-radius, radius + 1, dtype=np.float32)
    xx: np.ndarray
    yy: np.ndarray
    xx, yy = np.meshgrid(ax, ax)
    kernel: np.ndarray = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    kernel_sum: float = float(np.sum(kernel))
    if kernel_sum <= 0.0:
        raise ValueError("Gaussian kernel 和必須為正值。")
    return (kernel / kernel_sum).astype(np.float32)


def _infer_radius_from_sigma(sigma: float) -> int:
    """由 sigma 推估視窗半徑。"""
    if sigma <= 0:
        raise ValueError("sigma 必須為正值。")
    radius: int = int(math.ceil(3.0 * sigma))
    return max(radius, 1)


def _get_scipy_median_filter() -> Optional[Callable[..., np.ndarray]]:
    """取得 SciPy median_filter 函式（若可用）。"""
    global _SCIPY_MEDIAN_FILTER, _SCIPY_CHECKED
    if _SCIPY_CHECKED:
        return _SCIPY_MEDIAN_FILTER
    _SCIPY_CHECKED = True
    spec = importlib.util.find_spec("scipy.ndimage")
    if spec is None:
        return None
    module = importlib.import_module("scipy.ndimage")
    scipy_median_filter: Callable[..., np.ndarray] = getattr(module, "median_filter")
    _SCIPY_MEDIAN_FILTER = scipy_median_filter
    return _SCIPY_MEDIAN_FILTER


def _median_filter_naive(image: np.ndarray, radius: int) -> np.ndarray:
    """使用逐像素掃描的 median filter。"""
    height: int = image.shape[0]
    width: int = image.shape[1]
    window_size: int = radius * 2 + 1
    padded: np.ndarray = np.pad(image, ((radius, radius), (radius, radius)), mode="edge")
    output: np.ndarray = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            window: np.ndarray = padded[y : y + window_size, x : x + window_size]
            output[y, x] = float(np.median(window))
    return output


def _median_filter_vectorized(image: np.ndarray, radius: int, block_rows: int) -> np.ndarray:
    """使用滑動視窗與分塊策略加速 median filter。"""
    if block_rows <= 0:
        raise ValueError("block_rows 必須為正整數。")
    height: int = image.shape[0]
    width: int = image.shape[1]
    window_size: int = radius * 2 + 1
    padded: np.ndarray = np.pad(image, ((radius, radius), (radius, radius)), mode="edge")
    output: np.ndarray = np.zeros((height, width), dtype=np.float32)
    for row_start in range(0, height, block_rows):
        row_end: int = min(row_start + block_rows, height)
        chunk: np.ndarray = padded[row_start : row_end + 2 * radius, :]
        windows: np.ndarray = np.lib.stride_tricks.sliding_window_view(
            chunk, (window_size, window_size)
        )
        block_median: np.ndarray = np.median(windows, axis=(-1, -2))
        output[row_start:row_end, :] = block_median.astype(np.float32)
    return output


def _median_filter_scipy(image: np.ndarray, radius: int) -> np.ndarray:
    """使用 SciPy 的 median filter。"""
    scipy_filter: Optional[Callable[..., np.ndarray]] = _get_scipy_median_filter()
    if scipy_filter is None:
        raise ValueError("scipy 未安裝，無法使用 scipy 方法。")
    window_size: int = radius * 2 + 1
    filtered: np.ndarray = scipy_filter(image, size=window_size, mode="nearest")
    return filtered.astype(np.float32)


def median_filter(
    image: np.ndarray,
    radius: int,
    method: str = "auto",
    block_rows: int = 128,
) -> np.ndarray:
    """使用固定視窗半徑的 median filter。

    參數:
        image: 輸入 2D 影像。
        radius: 視窗半徑。
        method: 計算方法，"auto"、"scipy"、"vectorized" 或 "naive"。
        block_rows: 每次處理的列數，用於控制記憶體占用。

    回傳:
        濾波後影像，形狀為 (H, W)。
    """
    _validate_2d(image, "image")
    if radius <= 0:
        raise ValueError("radius 必須為正整數。")
    method_key: str = method.strip().lower()
    if method_key == "auto":
        if _get_scipy_median_filter() is not None:
            return _median_filter_scipy(image, radius)
        return _median_filter_vectorized(image, radius, block_rows)
    if method_key == "scipy":
        return _median_filter_scipy(image, radius)
    if method_key == "vectorized":
        return _median_filter_vectorized(image, radius, block_rows)
    if method_key == "naive":
        return _median_filter_naive(image, radius)
    raise ValueError("method 必須為 'auto'、'scipy'、'vectorized' 或 'naive'。")


def gaussian_filter(image: np.ndarray, sigma: float) -> np.ndarray:
    """使用 Gaussian filter 平滑影像。

    參數:
        image: 輸入 2D 影像。
        sigma: Gaussian 標準差。

    回傳:
        濾波後影像，形狀為 (H, W)。
    """
    _validate_2d(image, "image")
    radius: int = _infer_radius_from_sigma(sigma)
    kernel: np.ndarray = _gaussian_kernel(radius, sigma)
    height: int = image.shape[0]
    width: int = image.shape[1]
    padded: np.ndarray = np.pad(image, ((radius, radius), (radius, radius)), mode="reflect")
    output: np.ndarray = np.zeros((height, width), dtype=np.float32)
    size: int = radius * 2 + 1
    for y in range(height):
        for x in range(width):
            window: np.ndarray = padded[y : y + size, x : x + size]
            output[y, x] = float(np.sum(window * kernel))
    return output


def bilateral_filter(image: np.ndarray, sigma: float) -> np.ndarray:
    """使用 Bilateral filter 平滑影像並保留邊緣。

    參數:
        image: 輸入 2D 影像。
        sigma: 同時作為空間與強度的標準差。

    回傳:
        濾波後影像，形狀為 (H, W)。
    """
    _validate_2d(image, "image")
    radius: int = _infer_radius_from_sigma(sigma)
    spatial_kernel: np.ndarray = _gaussian_kernel(radius, sigma)
    height: int = image.shape[0]
    width: int = image.shape[1]
    padded: np.ndarray = np.pad(image, ((radius, radius), (radius, radius)), mode="reflect")
    output: np.ndarray = np.zeros((height, width), dtype=np.float32)
    size: int = radius * 2 + 1
    sigma_sq: float = float(sigma * sigma)
    for y in range(height):
        for x in range(width):
            window: np.ndarray = padded[y : y + size, x : x + size]
            center: float = float(padded[y + radius, x + radius])
            diff: np.ndarray = window - center
            range_weights: np.ndarray = np.exp(-(diff * diff) / (2.0 * sigma_sq))
            weights: np.ndarray = spatial_kernel * range_weights
            weight_sum: float = float(np.sum(weights))
            if weight_sum <= 0.0:
                output[y, x] = center
            else:
                output[y, x] = float(np.sum(window * weights) / weight_sum)
    return output
