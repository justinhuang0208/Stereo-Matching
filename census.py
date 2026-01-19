from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
try:
    import numba  # type: ignore[import-not-found]
    _NUMBA_AVAILABLE: bool = True
except ImportError:
    numba = None
    _NUMBA_AVAILABLE = False


def _get_numba_njit() -> Callable[[Callable[..., object]], Callable[..., object]]:
    """取得 Numba njit decorator，若不可用則回傳 no-op。"""
    if _NUMBA_AVAILABLE and numba is not None:
        return numba.njit(cache=True, fastmath=False)

    def _decorator(func: Callable[..., object]) -> Callable[..., object]:
        return func

    return _decorator


_NUMBA_NJIT: Callable[[Callable[..., object]], Callable[..., object]] = _get_numba_njit()


def _should_use_numba(use_numba: bool) -> bool:
    """判斷是否使用 Numba JIT。"""
    return bool(use_numba and _NUMBA_AVAILABLE)


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
    for dx, dy in directions:
        for r in range(1, radius + 1):
            offsets.append((dy * r, dx * r, r))
    return offsets


def compute_weights(offsets: Sequence[Tuple[int, int, int]], base_weight: float = 8.0) -> np.ndarray:
    """依距離生成權重，距離每增加 1 權重除以 2。

    參數:
        offsets: 位移清單，包含 (dy, dx, r)。
        base_weight: r=1 的基準權重。

    回傳:
        權重陣列，順序對應 offsets。
    """
    weights: List[float] = []
    for _, _, r in offsets:
        weight: float = base_weight / (2 ** (r - 1))
        weights.append(weight)
    return np.array(weights, dtype=np.float32)


def compute_valid_bounds(
    height: int,
    width: int,
    offsets: Sequence[Tuple[int, int, int]],
) -> Tuple[int, int, int, int]:
    """計算所有位移都有效的中心像素範圍。

    參數:
        height: 影像高度。
        width: 影像寬度。
        offsets: 位移清單，包含 (dy, dx, r)。

    回傳:
        (y_start, y_end, x_start, x_end) 的有效範圍。
    """
    y_start: int = 0
    y_end: int = height
    x_start: int = 0
    x_end: int = width
    for dy, dx, _ in offsets:
        y_start = max(y_start, max(0, -dy))
        y_end = min(y_end, height - max(0, dy))
        x_start = max(x_start, max(0, -dx))
        x_end = min(x_end, width - max(0, dx))
    return y_start, y_end, x_start, x_end


@_NUMBA_NJIT
def _compute_valid_bounds_numba(
    height: int,
    width: int,
    offsets: np.ndarray,
) -> Tuple[int, int, int, int]:
    """以 Numba 計算所有位移都有效的中心像素範圍。"""
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


@_NUMBA_NJIT
def _compute_census_bits_numba(
    image: np.ndarray,
    offsets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """以 Numba 計算 Census bits 與有效遮罩。"""
    height: int = int(image.shape[0])
    width: int = int(image.shape[1])
    num_offsets: int = int(offsets.shape[0])
    bits: np.ndarray = np.zeros((num_offsets, height, width), dtype=np.bool_)
    valid_mask: np.ndarray = np.zeros((height, width), dtype=np.bool_)
    y_start, y_end, x_start, x_end = _compute_valid_bounds_numba(height, width, offsets)
    if y_start < y_end and x_start < x_end:
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                valid_mask[y, x] = True
    for idx in range(num_offsets):
        dy: int = int(offsets[idx, 0])
        dx: int = int(offsets[idx, 1])
        y_src_start: int = 0 if dy >= 0 else -dy
        y_src_end: int = height - (dy if dy > 0 else 0)
        x_src_start: int = 0 if dx >= 0 else -dx
        x_src_end: int = width - (dx if dx > 0 else 0)
        if y_src_start >= y_src_end or x_src_start >= x_src_end:
            continue
        for y in range(y_src_start, y_src_end):
            y_nbr: int = y + dy
            for x in range(x_src_start, x_src_end):
                x_nbr: int = x + dx
                bits[idx, y, x] = image[y_nbr, x_nbr] > image[y, x]
    return bits, valid_mask


def compute_census_bits(
    image: np.ndarray,
    offsets: Sequence[Tuple[int, int, int]],
    use_numba: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """計算 Census bits 與有效像素遮罩。

    參數:
        image: 灰階影像陣列。
        offsets: 位移清單，包含 (dy, dx, r)。
        use_numba: 是否啟用 Numba JIT。

    回傳:
        (bits, valid_mask)
        bits: 形狀 (N, H, W) 的布林陣列。
        valid_mask: 形狀 (H, W) 的布林陣列，表示所有位移都有效的中心位置。
    """
    if image.ndim != 2:
        raise ValueError("image 必須為 2D 灰階影像。")
    if _should_use_numba(use_numba):
        offsets_array: np.ndarray = _offsets_to_array(offsets)
        return _compute_census_bits_numba(image, offsets_array)
    height: int = int(image.shape[0])
    width: int = int(image.shape[1])
    num_offsets: int = int(len(offsets))
    bits: np.ndarray = np.zeros((num_offsets, height, width), dtype=bool)

    y_start, y_end, x_start, x_end = compute_valid_bounds(height, width, offsets)
    valid_mask: np.ndarray = np.zeros((height, width), dtype=bool)
    if y_start < y_end and x_start < x_end:
        valid_mask[y_start:y_end, x_start:x_end] = True

    for idx, (dy, dx, _) in enumerate(offsets):
        y_src_start: int = max(0, -dy)
        y_src_end: int = height - max(0, dy)
        x_src_start: int = max(0, -dx)
        x_src_end: int = width - max(0, dx)
        if y_src_start >= y_src_end or x_src_start >= x_src_end:
            continue
        y_src = slice(y_src_start, y_src_end)
        x_src = slice(x_src_start, x_src_end)
        y_nbr = slice(y_src_start + dy, y_src_end + dy)
        x_nbr = slice(x_src_start + dx, x_src_end + dx)
        bits[idx, y_src, x_src] = image[y_nbr, x_nbr] > image[y_src, x_src]

    return bits, valid_mask


def compute_wct_cost_volume(
    left: np.ndarray,
    right: np.ndarray,
    dmax: int,
    radius: int = 4,
    base_weight: float = 8.0,
    use_numba: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> np.ndarray:
    """計算加權 Census Transform (WCT) 的 DSI cost volume。

    參數:
        left: 左影像灰階陣列。
        right: 右影像灰階陣列。
        dmax: 最大視差數量。
        radius: Census 半徑。
        base_weight: r=1 的基準權重。
        use_numba: 是否啟用 Numba JIT（僅套用到 bits 計算）。

    回傳:
        DSI cost volume，形狀為 (H, W, D)。
    """
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("left/right 必須為 2D 灰階影像。")
    if left.shape != right.shape:
        raise ValueError("left/right 影像尺寸不一致。")
    if dmax <= 0:
        raise ValueError("dmax 必須為正整數。")

    height: int = int(left.shape[0])
    width: int = int(left.shape[1])

    offsets: List[Tuple[int, int, int]] = generate_offsets(radius)
    weights: np.ndarray = compute_weights(offsets, base_weight)
    large_value: float = float(np.sum(weights))

    use_numba_bits: bool = _should_use_numba(use_numba)
    left_bits, left_valid = compute_census_bits(left, offsets, use_numba=use_numba_bits)
    right_bits, right_valid = compute_census_bits(right, offsets, use_numba=use_numba_bits)
    if left_valid.shape != right_valid.shape:
        raise ValueError("left/right 影像尺寸不一致。")

    dsi: np.ndarray = np.full((height, width, dmax), large_value, dtype=np.float32)
    weight_vector: np.ndarray = weights.astype(np.float32)

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
