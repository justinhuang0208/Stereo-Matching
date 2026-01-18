from __future__ import annotations

import argparse
import sys
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image

from census import compute_wct_cost_volume
from guided_filter import guided_filter
from stereo_io import ensure_same_shape, read_image, to_gray


def _print_progress(current: int, total: int, label: str) -> None:
    """在終端顯示簡單進度。

    參數:
        current: 目前完成數量。
        total: 總數量。
        label: 進度標籤。

    回傳:
        None。
    """
    if total <= 0:
        raise ValueError("total 必須為正整數。")
    clamped_current: int = min(max(current, 0), total)
    percent: float = (clamped_current / float(total)) * 100.0
    message: str = f"\r{label}: {clamped_current}/{total} ({percent:5.1f}%)"
    sys.stdout.write(message)
    sys.stdout.flush()


def aggregate_cost_volume(
    dsi: np.ndarray,
    guide: np.ndarray,
    radius: int,
    eps: float,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> np.ndarray:
    """使用 Guided Filter 對每個 disparity layer 做 cost aggregation。

    參數:
        dsi: 原始 cost volume，形狀為 (H, W, D)。
        guide: 引導影像（灰階）。
        radius: Guided Filter 視窗半徑。
        eps: 正則化項。

    回傳:
        聚合後 cost volume，形狀為 (H, W, D)。
    """
    if dsi.ndim != 3:
        raise ValueError("dsi 必須為 3D (H, W, D)。")
    if guide.ndim != 2:
        raise ValueError("guide 必須為 2D 灰階影像。")
    if dsi.shape[0] != guide.shape[0] or dsi.shape[1] != guide.shape[1]:
        raise ValueError("dsi 與 guide 尺寸不一致。")

    height: int = dsi.shape[0]
    width: int = dsi.shape[1]
    dmax: int = dsi.shape[2]
    aggregated: np.ndarray = np.zeros((height, width, dmax), dtype=np.float32)

    for d in range(dmax):
        aggregated[:, :, d] = guided_filter(guide, dsi[:, :, d], radius, eps)
        if progress_callback is not None:
            progress_callback(d + 1, dmax, "Guided Filter")

    if progress_callback is not None:
        sys.stdout.write("\n")

    return aggregated


def winner_take_all(cost_volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """沿 disparity 維度取最小 cost，回傳視差與最小 cost。

    參數:
        cost_volume: 聚合後 cost volume，形狀為 (H, W, D)。

    回傳:
        (disparity, min_cost)。
    """
    if cost_volume.ndim != 3:
        raise ValueError("cost_volume 必須為 3D (H, W, D)。")
    disparity: np.ndarray = np.argmin(cost_volume, axis=2).astype(np.int32)
    min_cost: np.ndarray = np.min(cost_volume, axis=2).astype(np.float32)
    return disparity, min_cost


def compute_disparity(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    dmax: int,
    wct_radius: int = 4,
    base_weight: float = 8.0,
    guided_radius: int = 3,
    guided_eps: float = 1e-3,
    show_progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """完整流程：WCT cost volume -> Guided Filter 聚合 -> WTA。

    參數:
        left_gray: 左影像灰階陣列。
        right_gray: 右影像灰階陣列。
        dmax: 最大視差數量。
        wct_radius: WCT 半徑。
        base_weight: WCT 基準權重。
        guided_radius: Guided Filter 半徑。
        guided_eps: Guided Filter 正則化項。

    回傳:
        (disparity, min_cost)。
    """
    ensure_same_shape(left_gray, right_gray)
    progress_callback: Optional[Callable[[int, int, str], None]] = _print_progress if show_progress else None
    if show_progress:
        print("WCT cost volume: 開始")
    dsi: np.ndarray = compute_wct_cost_volume(
        left_gray,
        right_gray,
        dmax=dmax,
        radius=wct_radius,
        base_weight=base_weight,
        progress_callback=progress_callback,
    )
    if show_progress:
        sys.stdout.write("\n")
        print("WCT cost volume: 完成")
    aggregated: np.ndarray = aggregate_cost_volume(
        dsi,
        left_gray,
        guided_radius,
        guided_eps,
        progress_callback=progress_callback,
    )
    disparity, min_cost = winner_take_all(aggregated)
    return disparity, min_cost


def _save_disparity_image(disparity: np.ndarray, dmax: int, path: str) -> None:
    """將視差圖正規化到 0~255 並輸出。

    參數:
        disparity: 視差圖。
        dmax: 最大視差數量。
        path: 輸出檔案路徑。

    回傳:
        None。
    """
    if dmax <= 0:
        raise ValueError("dmax 必須為正整數。")
    disp_norm: np.ndarray = (disparity.astype(np.float32) / float(dmax - 1)) * 255.0
    disp_img: Image.Image = Image.fromarray(disp_norm.astype(np.uint8), mode="L")
    disp_img.save(path)


def _parse_args() -> argparse.Namespace:
    """解析 CLI 參數。"""
    parser = argparse.ArgumentParser(description="Stereo Matching (WCT + Guided Filter + WTA)")
    parser.add_argument("--left", required=True, type=str, help="左影像路徑")
    parser.add_argument("--right", required=True, type=str, help="右影像路徑")
    parser.add_argument("--dmax", required=True, type=int, help="最大視差")
    parser.add_argument("--wct_radius", type=int, default=4, help="WCT 半徑")
    parser.add_argument("--base_weight", type=float, default=8.0, help="WCT 基準權重")
    parser.add_argument("--guided_radius", type=int, default=3, help="Guided Filter 半徑")
    parser.add_argument("--guided_eps", type=float, default=1e-3, help="Guided Filter epsilon")
    parser.add_argument("--output", type=str, default=None, help="輸出視差圖路徑")
    parser.add_argument("--output_npy", type=str, default=None, help="輸出視差 npy 路徑")
    parser.add_argument("--progress", action="store_true", help="顯示簡單進度")
    return parser.parse_args()


def main() -> None:
    """簡易驗證入口：讀入左右影像並輸出視差圖。"""
    args: argparse.Namespace = _parse_args()
    left_img: np.ndarray = read_image(args.left)
    right_img: np.ndarray = read_image(args.right)
    left_gray: np.ndarray = to_gray(left_img)
    right_gray: np.ndarray = to_gray(right_img)

    disparity, min_cost = compute_disparity(
        left_gray,
        right_gray,
        dmax=args.dmax,
        wct_radius=args.wct_radius,
        base_weight=args.base_weight,
        guided_radius=args.guided_radius,
        guided_eps=args.guided_eps,
        show_progress=args.progress,
    )

    if args.output is not None:
        _save_disparity_image(disparity, args.dmax, args.output)
    if args.output_npy is not None:
        np.save(args.output_npy, disparity)

    _ = min_cost


if __name__ == "__main__":
    main()
