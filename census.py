from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import numba


_NUMBA_NJIT: Callable[[Callable[..., object]], Callable[..., object]] = numba.njit(
    cache=True,
    fastmath=False,
)
_NUMBA_NJIT_PARALLEL: Callable[[Callable[..., object]], Callable[..., object]] = numba.njit(
    cache=True,
    fastmath=False,
    parallel=True,
)


def generate_offsets(radius: int = 4) -> List[Tuple[int, int, int]]:
    """產生 8 方向、距離 1..radius 的位移清單，也就是一個 window 中會被計算的部分。

    參數:
        radius: 位移最大距離。

    回傳:
        位移清單，每個元素為 (dy, dx, r)。
    """
    if radius <= 0:
        raise ValueError("radius 必須為正整數。")
    directions: Sequence[Tuple[int, int]] = (
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    )
    offsets: List[Tuple[int, int, int]] = []
    for dy, dx in directions:
        for r in range(1, radius + 1):
            offsets.append((dy * r, dx * r, r))
    return offsets


def compute_weights(offsets: Sequence[Tuple[int, int, int]], base_weight: float = 8.0) -> np.ndarray:
    """依距離生成權重，距離每增加 1 權重除以 2。

    參數:
        offsets: 位移陣列，形狀為 (N, 3) 的整數位移 (dy, dx, r)。
        base_weight: r=1 的基準權重。

    回傳:
        權重陣列，順序對應 offsets。
    """
    weights: List[float] = []
    for _, _, r in offsets:
        weight: float = base_weight / (2 ** (r - 1))
        weights.append(weight)
    return np.array(weights, dtype=np.float32)


@_NUMBA_NJIT
def _compute_valid_bounds_numba(
    height: int,
    width: int,
    offsets: np.ndarray,
) -> Tuple[int, int, int, int]:
    """以 Numba 計算所有位移都有效的中心像素範圍，確保在此範圍內的任何像素作為中心點時，其對應的所有鄰居像素都在影像內部。

    Args:
        height: 影像高度（像素數）。
        width: 影像寬度（像素數）。
        offsets: 位移陣列，形狀為 (N, 3) 的整數位移 (dy, dx, r)。

    Returns:
        y_start: 有效中心像素的起始列索引（含）。
        y_end: 有效中心像素的結束列索引（不含）。
        x_start: 有效中心像素的起始行索引（含）。
        x_end: 有效中心像素的結束行索引（不含）。
    """
    y_start: int = 0
    y_end: int = int(height)
    x_start: int = 0
    x_end: int = int(width)
    for i in range(offsets.shape[0]):
        dy: int = int(offsets[i, 0])
        dx: int = int(offsets[i, 1])
        if -dy > y_start:
            y_start = max(0, -dy)
        if dy > 0:
            y_end = min(y_end, height - dy)
        if -dx > x_start:
            x_start = max(0, -dx)
        if dx > 0:
            x_end = min(x_end, width - dx)
    return y_start, y_end, x_start, x_end


def _offsets_to_array(offsets: Sequence[Tuple[int, int, int]]) -> np.ndarray:
    """將 offsets 轉為 Numba 友善的整數陣列。"""
    return np.asarray(offsets, dtype=np.int32)


@_NUMBA_NJIT_PARALLEL
def compute_census_bits_numba(
    image: np.ndarray,
    offsets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """以 Numba 計算 Census bits 與有效遮罩。

    Args:
        image: 單通道影像，形狀為 (H, W)。
        offsets: 位移陣列，形狀為 (N, 3) 的整數位移 (dy, dx, r)。

    Returns:
        bits: Census bits，形狀為 (N, H, W) 的布林陣列。
        valid_mask: 中心像素有效遮罩，形狀為 (H, W) 的布林陣列。
    """
    height: int = int(image.shape[0])
    width: int = int(image.shape[1])
    num_offsets: int = int(offsets.shape[0])
    bits: np.ndarray = np.zeros((num_offsets, height, width), dtype=np.bool_)
    valid_mask: np.ndarray = np.zeros((height, width), dtype=np.bool_)
    # 先計算所有位移都有效的中心像素範圍
    y_start, y_end, x_start, x_end = _compute_valid_bounds_numba(height, width, offsets)
    if y_start < y_end and x_start < x_end:
        # 標記中心像素的有效區域，避免邊界越界
        for y in numba.prange(y_start, y_end):
            for x in range(x_start, x_end):
                valid_mask[y, x] = True
    # 逐位移比較鄰點與中心點，建立 Census bits
    for idx in numba.prange(num_offsets):
        dy: int = int(offsets[idx, 0])
        dx: int = int(offsets[idx, 1])
        # 計算有效的中心像素範圍，確保鄰居點 (y+dy, x+dx) 不會超出影像邊界
        # 當 dy < 0（向上移動）時，中心點 y 必須從 -dy 開始，否則鄰居會超出上邊界
        y_src_start: int = -dy if dy < 0 else 0
        # 當 dy > 0（向下移動）時，中心點 y 最大只能到 height - dy，否則鄰居會超出下邊界
        y_src_end: int = height - dy if dy > 0 else height
        # x 方向的邏輯與 y 方向相同
        x_src_start: int = -dx if dx < 0 else 0
        x_src_end: int = width - dx if dx > 0 else width
        if y_src_start >= y_src_end or x_src_start >= x_src_end:
            continue
        for y in range(y_src_start, y_src_end):
            y_nbr: int = y + dy
            for x in range(x_src_start, x_src_end):
                x_nbr: int = x + dx
                bits[idx, y, x] = image[y_nbr, x_nbr] > image[y, x]
    return bits, valid_mask


@_NUMBA_NJIT_PARALLEL
def _compute_wct_cost_volume_parallel_range(
    left_bits: np.ndarray,
    right_bits: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
    weights: np.ndarray,
    dsi: np.ndarray,
    d_start: int,
    d_end: int,
    large_value: float,
) -> None:
    """以 Numba 平行化計算指定視差範圍內的 WCT cost volume。

    Args:
        left_bits: 左影像 Census bits，形狀為 (N, H, W)。
        right_bits: 右影像 Census bits，形狀為 (N, H, W)。
        left_valid: 左影像有效遮罩，形狀為 (H, W)。
        right_valid: 右影像有效遮罩，形狀為 (H, W)。
        weights: 權重陣列，形狀為 (N,)。
        dsi: 要寫入的 cost volume，形狀為 (H, W, D)。
        d_start: 視差起始（含）。
        d_end: 視差結束（不含）。
        large_value: 無效區域的成本值。

    Returns:
        None。
    """
    num_offsets: int = int(left_bits.shape[0])
    height: int = int(left_bits.shape[1])
    width: int = int(left_bits.shape[2])
    for d in numba.prange(d_start, d_end):
        for y in range(height):
            for x in range(width):
                # 判斷視差 d 在右影像中是否超出了左邊界
                if d > 0 and x < d:
                    dsi[y, x, d] = large_value
                    continue
                xr: int = x - d
                if not (left_valid[y, x] and right_valid[y, xr]):
                    dsi[y, x, d] = large_value
                    continue
                cost: np.float32 = np.float32(0.0)
                for idx in range(num_offsets):
                    if left_bits[idx, y, x] != right_bits[idx, y, xr]:
                        cost = np.float32(cost + weights[idx])
                dsi[y, x, d] = cost


def compute_wct_cost_volume(
    left: np.ndarray,
    right: np.ndarray,
    dmax: int,
    radius: int = 4,
    base_weight: float = 8.0,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    parallel: bool = True,
    parallel_chunk: int = 8,
) -> np.ndarray:
    """計算加權 Census Transform (WCT) 的 DSI cost volume。

    參數:
        left: 左影像灰階陣列。
        right: 右影像灰階陣列。
        dmax: 最大視差數量。
        radius: Census 半徑。
        base_weight: r=1 的基準權重。
        parallel: 是否啟用 Numba 平行化路徑。
        parallel_chunk: 平行化時每次處理的視差數量。
    回傳:
        DSI cost volume，形狀為 (H, W, D)。
    """
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("left/right 必須為 2D 灰階影像。")
    if left.shape != right.shape:
        raise ValueError("left/right 影像尺寸不一致。")
    if dmax <= 0:
        raise ValueError("dmax 必須為正整數。")
    if parallel_chunk <= 0:
        raise ValueError("parallel_chunk 必須為正整數。")

    height: int = int(left.shape[0])
    width: int = int(left.shape[1])

    offsets: List[Tuple[int, int, int]] = generate_offsets(radius)
    weights: np.ndarray = compute_weights(offsets, base_weight)
    large_value: float = float(np.sum(weights)) * 10.0
    offsets_array: np.ndarray = _offsets_to_array(offsets)

    # 計算左右影像的 Census bits 與有效遮罩
    left_bits, left_valid = compute_census_bits_numba(left, offsets_array)
    right_bits, right_valid = compute_census_bits_numba(right, offsets_array)
    if left_valid.shape != right_valid.shape:
        raise ValueError("left/right 影像尺寸不一致。")

    # 初始化 DSI cost volume，預設為最大成本
    dsi: np.ndarray = np.full((height, width, dmax), large_value, dtype=np.float32)
    weight_vector: np.ndarray = weights.astype(np.float32)

    if parallel:
        for d_start in range(0, dmax, parallel_chunk):
            d_end: int = min(d_start + parallel_chunk, dmax)
            _compute_wct_cost_volume_parallel_range(
                left_bits,
                right_bits,
                left_valid,
                right_valid,
                weight_vector,
                dsi,
                d_start,
                d_end,
                float(large_value),
            )
            if progress_callback is not None:
                progress_callback(d_end, dmax, "WCT cost volume")
        return dsi

    # 逐視差計算 WCT cost，並依有效遮罩填入
    for d in range(dmax):
        if d >= width:
            if progress_callback is not None:
                progress_callback(d + 1, dmax, "WCT cost volume")
            continue
        if d == 0:
            xor_bits: np.ndarray = left_bits ^ right_bits
            cost_slice: np.ndarray = np.tensordot(
                weight_vector,
                xor_bits.astype(np.float32),
                axes=(0, 0),
            )
            valid_slice: np.ndarray = left_valid & right_valid
            dsi[:, :, d] = np.where(valid_slice, cost_slice, large_value)
        else:
            xor_bits = left_bits[:, :, d:] ^ right_bits[:, :, :-d]
            cost_slice = np.tensordot(
                weight_vector,
                xor_bits.astype(np.float32),
                axes=(0, 0),
            )
            valid_slice = left_valid[:, d:] & right_valid[:, :-d]
            dsi[:, d:, d] = np.where(valid_slice, cost_slice, large_value)
        if progress_callback is not None:
            progress_callback(d + 1, dmax, "WCT cost volume")

    return dsi
