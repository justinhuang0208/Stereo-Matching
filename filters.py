from __future__ import annotations

import math
from typing import Tuple

import numpy as np


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


def median_filter(image: np.ndarray, radius: int) -> np.ndarray:
    """使用固定視窗半徑的 median filter。

    參數:
        image: 輸入 2D 影像。
        radius: 視窗半徑。

    回傳:
        濾波後影像，形狀為 (H, W)。
    """
    _validate_2d(image, "image")
    if radius <= 0:
        raise ValueError("radius 必須為正整數。")
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
