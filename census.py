from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


def generate_offsets(radius: int = 4) -> List[Tuple[int, int, int]]:
    """產生 8 方向、距離 1..radius 的位移清單。

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


def compute_wct_cost_volume(
    left: np.ndarray,
    right: np.ndarray,
    dmax: int,
    radius: int = 4,
    base_weight: float = 8.0,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> np.ndarray:
    """計算加權 Census Transform (WCT) 的 DSI cost volume。

    參數:
        left: 左影像灰階陣列。
        right: 右影像灰階陣列。
        dmax: 最大視差數量。
        radius: Census 半徑。
        base_weight: r=1 的基準權重。

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

    dsi: np.ndarray = np.full((height, width, dmax), large_value, dtype=np.float32)

    for y in range(height):
        for x in range(width):
            left_center: float = float(left[y, x])
            for d in range(dmax):
                xr: int = x - d
                if xr < 0:
                    continue
                right_center: float = float(right[y, xr])
                cost: float = 0.0
                valid: bool = True
                for (dy, dx, _), weight in zip(offsets, weights):
                    yl: int = y + dy
                    xl: int = x + dx
                    yr: int = y + dy
                    xr2: int = xr + dx
                    if (
                        yl < 0
                        or yl >= height
                        or xl < 0
                        or xl >= width
                        or yr < 0
                        or yr >= height
                        or xr2 < 0
                        or xr2 >= width
                    ):
                        valid = False
                        break
                    left_bit: bool = bool(left[yl, xl] > left_center)
                    right_bit: bool = bool(right[yr, xr2] > right_center)
                    if left_bit != right_bit:
                        cost += float(weight)
                if valid:
                    dsi[y, x, d] = np.float32(cost)
        if progress_callback is not None:
            progress_callback(y + 1, height, "WCT cost volume")

    return dsi
