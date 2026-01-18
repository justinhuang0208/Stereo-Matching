from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from census import compute_wct_cost_volume
from filters import bilateral_filter, gaussian_filter, median_filter
from guided_filter import guided_filter
from stereo_io import ensure_same_shape, read_image, to_gray

DEFAULT_WCT_RADIUS: int = 4
DEFAULT_BASE_WEIGHT: float = 8.0
DEFAULT_GUIDED_RADIUS: int = 3
DEFAULT_GUIDED_EPS: float = 0.0154
DEFAULT_FILTER_TYPE: str = "guided"
DEFAULT_MEDIAN_RADIUS: int = 3
DEFAULT_MEDIAN_METHOD: str = "auto"
DEFAULT_MEDIAN_BLOCK_ROWS: int = 128
DEFAULT_GAUSSIAN_SIGMA: float = 1.0
DEFAULT_BILATERAL_SIGMA: float = 1.0


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
    guided_radius: int,
    guided_eps: float,
    filter_type: str = DEFAULT_FILTER_TYPE,
    median_radius: int = DEFAULT_MEDIAN_RADIUS,
    median_method: str = DEFAULT_MEDIAN_METHOD,
    median_block_rows: int = DEFAULT_MEDIAN_BLOCK_ROWS,
    gaussian_sigma: float = DEFAULT_GAUSSIAN_SIGMA,
    bilateral_sigma: float = DEFAULT_BILATERAL_SIGMA,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> np.ndarray:
    """使用指定的濾波器對每個 disparity layer 做 cost aggregation。

    參數:
        dsi: 原始 cost volume，形狀為 (H, W, D)。
        guide: 引導影像（灰階）。
        guided_radius: Guided Filter 視窗半徑。
        guided_eps: 正則化項。
        filter_type: 聚合濾波器類型（guided, median, gaussian, bilateral）。
        median_radius: Median Filter 視窗半徑。
        median_method: Median Filter 計算方法（auto, scipy, vectorized, naive）。
        median_block_rows: Median Filter 分塊列數。
        gaussian_sigma: Gaussian Filter 標準差。
        bilateral_sigma: Bilateral Filter 標準差。

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
    filter_key: str = filter_type.strip().lower()
    supported: Tuple[str, ...] = ("guided", "median", "gaussian", "bilateral")
    if filter_key not in supported:
        raise ValueError(f"filter_type 必須為 {supported} 其中之一。")

    for d in range(dmax):
        layer: np.ndarray = dsi[:, :, d]
        if filter_key == "guided":
            filtered: np.ndarray = guided_filter(guide, layer, guided_radius, guided_eps)
            label: str = "Guided Filter"
        elif filter_key == "median":
            filtered = median_filter(
                layer,
                median_radius,
                method=median_method,
                block_rows=median_block_rows,
            )
            label = "Median Filter"
        elif filter_key == "gaussian":
            filtered = gaussian_filter(layer, gaussian_sigma)
            label = "Gaussian Filter"
        else:
            filtered = bilateral_filter(layer, bilateral_sigma)
            label = "Bilateral Filter"
        aggregated[:, :, d] = filtered
        if progress_callback is not None:
            progress_callback(d + 1, dmax, label)

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
    wct_radius: int = DEFAULT_WCT_RADIUS,
    base_weight: float = DEFAULT_BASE_WEIGHT,
    guided_radius: int = DEFAULT_GUIDED_RADIUS,
    guided_eps: float = DEFAULT_GUIDED_EPS,
    filter_type: str = DEFAULT_FILTER_TYPE,
    median_radius: int = DEFAULT_MEDIAN_RADIUS,
    median_method: str = DEFAULT_MEDIAN_METHOD,
    median_block_rows: int = DEFAULT_MEDIAN_BLOCK_ROWS,
    gaussian_sigma: float = DEFAULT_GAUSSIAN_SIGMA,
    bilateral_sigma: float = DEFAULT_BILATERAL_SIGMA,
    show_progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """完整流程：WCT cost volume -> 濾波聚合 -> WTA。

    參數:
        left_gray: 左影像灰階陣列。
        right_gray: 右影像灰階陣列。
        dmax: 最大視差數量。
        wct_radius: WCT 半徑。
        base_weight: WCT 基準權重。
        guided_radius: Guided Filter 半徑。
        guided_eps: Guided Filter 正則化項。
        filter_type: 聚合濾波器類型（guided, median, gaussian, bilateral）。
        median_radius: Median Filter 視窗半徑。
        median_method: Median Filter 計算方法（auto, scipy, vectorized, naive）。
        median_block_rows: Median Filter 分塊列數。
        gaussian_sigma: Gaussian Filter 標準差。
        bilateral_sigma: Bilateral Filter 標準差。

    回傳:
        (disparity, min_cost)。
    """
    ensure_same_shape(left_gray, right_gray)
    progress_callback: Optional[Callable[[int, int, str], None]] = _print_progress if show_progress else None
    dsi: np.ndarray = compute_wct_cost_volume(
        left_gray,
        right_gray,
        dmax=dmax,
        radius=wct_radius,
        base_weight=base_weight,
        progress_callback=progress_callback,
    )
    aggregated: np.ndarray = aggregate_cost_volume(
        dsi,
        left_gray,
        guided_radius,
        guided_eps,
        filter_type=filter_type,
        median_radius=median_radius,
        median_method=median_method,
        median_block_rows=median_block_rows,
        gaussian_sigma=gaussian_sigma,
        bilateral_sigma=bilateral_sigma,
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


def _jet_colormap(values: np.ndarray) -> np.ndarray:
    """將 0~1 的數值映射為 Jet 顏色。

    參數:
        values: 0~1 的正規化陣列。

    回傳:
        RGB 顏色陣列，形狀為 (..., 3)，範圍 0~1。
    """
    if values.ndim < 2:
        raise ValueError("values 必須至少為 2D。")
    v: np.ndarray = np.clip(values.astype(np.float32), 0.0, 1.0)
    four_v: np.ndarray = 4.0 * v
    r: np.ndarray = np.clip(np.minimum(four_v - 1.5, -four_v + 4.5), 0.0, 1.0)
    g: np.ndarray = np.clip(np.minimum(four_v - 0.5, -four_v + 3.5), 0.0, 1.0)
    b: np.ndarray = np.clip(np.minimum(four_v + 0.5, -four_v + 2.5), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def _save_disparity_color_image(disparity: np.ndarray, dmax: int, path: str) -> None:
    """將視差圖以 Jet 色盤輸出成彩色圖片。

    參數:
        disparity: 視差圖。
        dmax: 最大視差數量。
        path: 輸出檔案路徑。

    回傳:
        None。
    """
    if dmax <= 0:
        raise ValueError("dmax 必須為正整數。")
    disp_norm: np.ndarray = disparity.astype(np.float32) / float(dmax - 1)
    rgb: np.ndarray = _jet_colormap(disp_norm) * 255.0
    disp_img: Image.Image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    disp_img.save(path)


def _parse_args() -> argparse.Namespace:
    """解析 CLI 參數。"""
    parser = argparse.ArgumentParser(description="Stereo Matching (WCT + Guided Filter + WTA)")
    parser.add_argument("--left", required=True, type=str, help="左影像路徑")
    parser.add_argument("--right", required=True, type=str, help="右影像路徑")
    parser.add_argument("--dmax", required=True, type=int, help="最大視差")
    parser.add_argument("--wct_radius", type=int, default=DEFAULT_WCT_RADIUS, help="WCT 半徑")
    parser.add_argument("--base_weight", type=float, default=DEFAULT_BASE_WEIGHT, help="WCT 基準權重")
    parser.add_argument("--guided_radius", type=int, default=DEFAULT_GUIDED_RADIUS, help="Guided Filter 半徑")
    parser.add_argument("--guided_eps", type=float, default=DEFAULT_GUIDED_EPS, help="Guided Filter epsilon")
    parser.add_argument(
        "--filter",
        type=str,
        default=DEFAULT_FILTER_TYPE,
        choices=["guided", "median", "gaussian", "bilateral"],
        help="聚合濾波器類型",
    )
    parser.add_argument("--median_radius", type=int, default=DEFAULT_MEDIAN_RADIUS, help="Median Filter 半徑")
    parser.add_argument(
        "--median_method",
        type=str,
        default=DEFAULT_MEDIAN_METHOD,
        choices=["auto", "scipy", "vectorized", "naive"],
        help="Median Filter 計算方法",
    )
    parser.add_argument(
        "--median_block_rows",
        type=int,
        default=DEFAULT_MEDIAN_BLOCK_ROWS,
        help="Median Filter 分塊列數",
    )
    parser.add_argument("--gaussian_sigma", type=float, default=DEFAULT_GAUSSIAN_SIGMA, help="Gaussian sigma")
    parser.add_argument("--bilateral_sigma", type=float, default=DEFAULT_BILATERAL_SIGMA, help="Bilateral sigma")
    parser.add_argument("--output", type=str, default=None, help="輸出視差圖路徑")
    parser.add_argument("--output_color", type=str, default=None, help="輸出彩色視差圖路徑")
    parser.add_argument("--output_npy", type=str, default=None, help="輸出視差 npy 路徑")
    parser.add_argument("--progress", action="store_true", help="顯示簡單進度")
    return parser.parse_args()


def _create_run_directory(base_dir: str, timestamp: str) -> Path:
    """建立本次執行的輸出資料夾。

    參數:
        base_dir: 輸出根資料夾。
        timestamp: 時間字串（YYYYMMDDHHMM）。

    回傳:
        本次執行資料夾 Path。
    """
    root: Path = Path(base_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_dir: Path = root / timestamp
    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir
    suffix: int = 1
    while True:
        candidate: Path = root / f"{timestamp}_{suffix:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        suffix += 1


def _build_run_metadata(
    args: argparse.Namespace,
    run_dir: Path,
    output_gray: Path,
    output_color: Path,
) -> Dict[str, str]:
    """建立本次執行的參數摘要。

    參數:
        args: CLI 參數。
        run_dir: 本次執行資料夾。
        output_gray: 灰階視差圖輸出路徑。
        output_color: 彩色視差圖輸出路徑。

    回傳:
        參數摘要字典。
    """
    return {
        "timestamp": run_dir.name,
        "run_dir": str(run_dir),
        "left": str(args.left),
        "right": str(args.right),
        "dmax": str(args.dmax),
        "wct_radius": str(args.wct_radius),
        "base_weight": str(args.base_weight),
        "guided_radius": str(args.guided_radius),
        "guided_eps": str(args.guided_eps),
        "filter": str(args.filter),
        "median_radius": str(args.median_radius),
        "median_method": str(args.median_method),
        "median_block_rows": str(args.median_block_rows),
        "gaussian_sigma": str(args.gaussian_sigma),
        "bilateral_sigma": str(args.bilateral_sigma),
        "progress": str(bool(args.progress)),
        "output_disparity_png": str(output_gray),
        "output_disparity_color_png": str(output_color),
        "output_arg_gray": str(args.output) if args.output is not None else "",
        "output_arg_color": str(args.output_color) if args.output_color is not None else "",
        "output_arg_npy": str(args.output_npy) if args.output_npy is not None else "",
    }


def _write_run_metadata(path: Path, metadata: Dict[str, str]) -> None:
    """輸出本次執行參數檔案。

    參數:
        path: 輸出檔案路徑。
        metadata: 參數摘要。

    回傳:
        None。
    """
    with path.open("w", encoding="utf-8") as handler:
        json.dump(metadata, handler, ensure_ascii=True, indent=2, sort_keys=True)


def main() -> None:
    """簡易驗證入口：讀入左右影像並輸出視差圖。"""
    args: argparse.Namespace = _parse_args()
    timestamp: str = datetime.now().strftime("%Y%m%d%H%M")
    run_dir: Path = _create_run_directory("result", timestamp)
    output_gray: Path = run_dir / "disparity.png"
    output_color: Path = run_dir / "disparity_color.png"
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
        filter_type=args.filter,
        median_radius=args.median_radius,
        median_method=args.median_method,
        median_block_rows=args.median_block_rows,
        gaussian_sigma=args.gaussian_sigma,
        bilateral_sigma=args.bilateral_sigma,
        show_progress=args.progress,
    )

    _save_disparity_image(disparity, args.dmax, str(output_gray))
    _save_disparity_color_image(disparity, args.dmax, str(output_color))
    metadata: Dict[str, str] = _build_run_metadata(args, run_dir, output_gray, output_color)
    _write_run_metadata(run_dir / "params.json", metadata)

    if args.output is not None:
        _save_disparity_image(disparity, args.dmax, args.output)
    if args.output_color is not None:
        _save_disparity_color_image(disparity, args.dmax, args.output_color)
    if args.output_npy is not None:
        np.save(args.output_npy, disparity)

    _ = min_cost


if __name__ == "__main__":
    main()
