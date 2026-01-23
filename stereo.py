from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from census import compute_wct_cost_volume
from filters import bilateral_filter, gaussian_filter, median_filter
from guided_filter import GuidedFilterPrecomputed, guided_filter_with_precompute, prepare_guided_filter
from stereo_io import ensure_same_shape, read_image, read_pfm, read_pgm_mask, save_disparity_npz, to_gray

DEFAULT_WCT_RADIUS: int = 4
DEFAULT_BASE_WEIGHT: float = 8.0
DEFAULT_GUIDED_RADIUS: int = 3
DEFAULT_GUIDED_EPS: float = 1000
DEFAULT_FILTER_TYPE: str = "guided"
DEFAULT_MEDIAN_RADIUS: int = 3
DEFAULT_MEDIAN_METHOD: str = "auto"
DEFAULT_MEDIAN_BLOCK_ROWS: int = 128
DEFAULT_GAUSSIAN_SIGMA: float = 1.0
DEFAULT_BILATERAL_SIGMA: float = 1.0
DEFAULT_BAD_THRESHOLD: float = 2.0


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
    message: str = f"{label}: {clamped_current}/{total} ({percent:5.1f}%)"
    if clamped_current >= total:
        sys.stdout.write(f"\r{message}\n")
    else:
        sys.stdout.write(f"\r{message}")
    sys.stdout.flush()


@dataclass
class DatasetProgressState:
    """批次處理 dataset 的進度顯示狀態。"""

    rendered: bool = False


def _print_dataset_stage_progress(
    dataset_current: int,
    dataset_total: int,
    stage_current: int,
    stage_total: int,
    label: str,
    state: DatasetProgressState,
) -> None:
    """顯示 dataset 與當前步驟的雙行進度。"""
    if dataset_total <= 0:
        raise ValueError("dataset_total 必須為正整數。")
    if stage_total <= 0:
        raise ValueError("stage_total 必須為正整數。")
    clamped_dataset: int = min(max(dataset_current, 0), dataset_total)
    clamped_stage: int = min(max(stage_current, 0), stage_total)
    dataset_percent: float = (clamped_dataset / float(dataset_total)) * 100.0
    stage_percent: float = (clamped_stage / float(stage_total)) * 100.0
    dataset_line: str = f"Dataset: {clamped_dataset}/{dataset_total} ({dataset_percent:5.1f}%)"
    stage_line: str = f"{label}: {clamped_stage}/{stage_total} ({stage_percent:5.1f}%)"
    if not state.rendered:
        sys.stdout.write(f"{dataset_line}\n{stage_line}")
        sys.stdout.flush()
        state.rendered = True
        return
    sys.stdout.write("\033[1A\r\033[2K")
    sys.stdout.write(dataset_line)
    sys.stdout.write("\n\r\033[2K")
    sys.stdout.write(stage_line)
    sys.stdout.flush()


def aggregate_and_wta(
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
) -> Tuple[np.ndarray, np.ndarray]:
    """逐層聚合並即時更新 WTA，避免建立完整聚合 cost volume。

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
        (disparity, min_cost)。
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
    min_cost: np.ndarray = np.full((height, width), np.inf, dtype=np.float32)
    disparity: np.ndarray = np.zeros((height, width), dtype=np.int32)
    filter_key: str = filter_type.strip().lower()
    supported: Tuple[str, ...] = ("guided", "median", "gaussian", "bilateral")
    if filter_key not in supported:
        raise ValueError(f"filter_type 必須為 {supported} 其中之一。")

    guided_cache: Optional[GuidedFilterPrecomputed] = None
    if filter_key == "guided":
        guided_cache = prepare_guided_filter(guide, guided_radius, guided_eps)

    for d in range(dmax):
        layer: np.ndarray = dsi[:, :, d]
        if filter_key == "guided":
            if guided_cache is None:
                raise RuntimeError("guided_cache 不可為 None。")
            filtered = guided_filter_with_precompute(guided_cache, layer)
            label = "Guided Filter"
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

        better: np.ndarray = filtered < min_cost
        if np.any(better):
            min_cost = np.where(better, filtered, min_cost)
            disparity[better] = d
        if progress_callback is not None:
            progress_callback(d + 1, dmax, label)

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
    show_progress: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
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
        show_progress: 是否顯示進度（progress_callback 有提供時會忽略）。
        progress_callback: 自訂進度回呼函式。

    回傳:
        (disparity, min_cost)。
    """
    ensure_same_shape(left_gray, right_gray)
    progress: Optional[Callable[[int, int, str], None]]
    if progress_callback is not None:
        progress = progress_callback
    elif show_progress:
        progress = _print_progress
    else:
        progress = None
    dsi: np.ndarray = compute_wct_cost_volume(
        left_gray,
        right_gray,
        dmax=dmax,
        radius=wct_radius,
        base_weight=base_weight,
        progress_callback=progress,
    )
    disparity, min_cost = aggregate_and_wta(
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
        progress_callback=progress,
    )
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


def _read_ndisp_summary(path: Path) -> Dict[str, int]:
    """讀取 ndisp_summary.csv 對應表。

    參數:
        path: CSV 檔案路徑。

    回傳:
        場景名稱到 ndisp 的映射。
    """
    if not path.exists():
        raise FileNotFoundError(f"找不到 ndisp_summary.csv: {path}")
    mapping: Dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handler:
        reader = csv.DictReader(handler)
        for row in reader:
            scene: str = (row.get("scene") or "").strip()
            ndisp_raw: str = (row.get("ndisp") or "").strip()
            if not scene or not ndisp_raw:
                continue
            if not scene.endswith("-perfect"):
                continue
            try:
                ndisp: int = int(ndisp_raw)
            except ValueError:
                continue
            mapping[scene] = ndisp
    if not mapping:
        raise ValueError("ndisp_summary.csv 內容為空或無有效場景。")
    return mapping


def _resolve_dataset_paths(dataset_root: Path, scene: str) -> Tuple[Path, Path, Path, Path]:
    """依場景名稱組合資料路徑。

    參數:
        dataset_root: dataset 根目錄。
        scene: 場景名稱。

    回傳:
        (left_path, right_path, gt_path, gt_mask_path)。
    """
    scene_dir: Path = dataset_root / scene
    left_path: Path = scene_dir / "im0.png"
    right_path: Path = scene_dir / "im1.png"
    gt_path: Path = scene_dir / "disp0.pfm"
    gt_mask_path: Path = scene_dir / "disp0-n.pgm"
    return left_path, right_path, gt_path, gt_mask_path


def _resolve_scene_inputs(
    dataset_root: Path,
    scene: str,
    ndisp_map: Dict[str, int],
    dmax_override: int,
) -> Tuple[Path, Path, Path, Path, int]:
    """解析場景對應輸入與 dmax。

    參數:
        dataset_root: dataset 根目錄。
        scene: 場景名稱。
        ndisp_map: 場景到 ndisp 的映射。
        dmax_override: 覆蓋用 dmax（<=0 表示不覆蓋）。

    回傳:
        (left_path, right_path, gt_path, gt_mask_path, dmax)。
    """
    left_path, right_path, gt_path, gt_mask_path = _resolve_dataset_paths(dataset_root, scene)
    if not left_path.exists():
        raise FileNotFoundError(f"找不到左影像: {left_path}")
    if not right_path.exists():
        raise FileNotFoundError(f"找不到右影像: {right_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"找不到 GT: {gt_path}")
    if not gt_mask_path.exists():
        raise FileNotFoundError(f"找不到 GT 遮罩: {gt_mask_path}")
    if dmax_override > 0:
        return left_path, right_path, gt_path, gt_mask_path, dmax_override
    if scene not in ndisp_map:
        raise ValueError(f"ndisp_summary.csv 缺少場景 {scene} 的 dmax，請手動指定 --dmax。")
    return left_path, right_path, gt_path, gt_mask_path, int(ndisp_map[scene])


def _parse_args() -> argparse.Namespace:
    """解析 CLI 參數。"""
    parser = argparse.ArgumentParser(description="Stereo Matching (WCT + Guided Filter + WTA)")
    parser.add_argument("--left", type=str, default="", help="左影像路徑")
    parser.add_argument("--right", type=str, default="", help="右影像路徑")
    parser.add_argument("--dmax", type=int, default=0, help="最大視差")
    parser.add_argument("--dataset", type=str, default="", help="dataset 資料夾名稱")
    parser.add_argument("--all-datasets", action="store_true", help="批次處理 dataset 內所有場景")
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
    parser.add_argument("--gt", type=str, default="", help="GT PFM 檔案路徑")
    parser.add_argument("--gt-mask", type=str, default="", help="GT 遮罩 PGM 檔案路徑")
    parser.add_argument("--bad_threshold", type=float, default=DEFAULT_BAD_THRESHOLD, help="Bad pixel 閾值")
    return parser.parse_args()


def _create_run_directory(base_dir: str, timestamp: str) -> Path:
    """建立本次執行的輸出資料夾。

    參數:
        base_dir: 輸出根資料夾。
        timestamp: 時間字串（YYYYMMDDHHMMSS）。
            若遇到同名資料夾，會在日期與時間之間插入序號，確保名稱最後六碼為 HHMMSS。

    回傳:
        本次執行資料夾 Path。
    """
    root: Path = Path(base_dir)
    root.mkdir(parents=True, exist_ok=True)
    date_part: str = timestamp[:-6]
    time_part: str = timestamp[-6:]
    run_dir: Path = root / f"{date_part}{time_part}"
    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir
    suffix: int = 1
    while True:
        if date_part:
            candidate_name: str = f"{date_part}_{suffix:02d}_{time_part}"
        else:
            candidate_name = f"{suffix:02d}_{time_part}"
        candidate: Path = root / candidate_name
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        suffix += 1


def _build_run_metadata(
    args: argparse.Namespace,
    run_dir: Path,
    output_gray: Path,
    output_color: Path,
    output_npz: Path,
    output_metrics: Optional[Path],
    resolved_left: Path,
    resolved_right: Path,
    resolved_gt: Path,
    resolved_gt_mask: Path,
    resolved_dmax: int,
    dataset_name: str,
    all_datasets: bool,
) -> Dict[str, str]:
    """建立本次執行的參數摘要。

    參數:
        args: CLI 參數。
        run_dir: 本次執行資料夾。
        output_gray: 灰階視差圖輸出路徑。
        output_color: 彩色視差圖輸出路徑。
        output_npz: 原始資料輸出路徑。
        output_metrics: 評估結果輸出路徑。
        resolved_gt_mask: GT 遮罩路徑。

    回傳:
        參數摘要字典。
    """
    return {
        "timestamp": run_dir.name,
        "run_dir": str(run_dir),
        "left": str(resolved_left),
        "right": str(resolved_right),
        "dmax": str(resolved_dmax),
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
        "bad_threshold": str(args.bad_threshold),
        "eval": str(True),
        "gt": str(resolved_gt),
        "gt_mask": str(resolved_gt_mask),
        "dataset": dataset_name,
        "all_datasets": str(bool(all_datasets)),
        "progress": str(True),
        "output_disparity_png": str(output_gray),
        "output_disparity_color_png": str(output_color),
        "output_disparity_npz": str(output_npz),
        "output_metrics_json": "" if output_metrics is None else str(output_metrics),
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


def _compute_pbm_rms(
    disparity: np.ndarray,
    ground_truth: np.ndarray,
    valid_mask: np.ndarray,
    bad_threshold: float,
) -> Dict[str, float]:
    """計算 PBM 與 RMS 指標。

    參數:
        disparity: 估計視差圖。
        ground_truth: GT 視差圖（PFM）。
        valid_mask: GT 有效像素遮罩。
        bad_threshold: bad pixel 閾值。

    回傳:
        包含 PBM 與 RMS 的字典。
    """
    if disparity.shape != ground_truth.shape:
        raise ValueError("disparity 與 ground_truth 尺寸不一致。")
    if disparity.ndim != 2 or ground_truth.ndim != 2:
        raise ValueError("disparity 與 ground_truth 必須為 2D。")
    if valid_mask.shape != ground_truth.shape:
        raise ValueError("valid_mask 與 ground_truth 尺寸不一致。")
    if valid_mask.ndim != 2:
        raise ValueError("valid_mask 必須為 2D。")
    if bad_threshold <= 0:
        raise ValueError("bad_threshold 必須為正數。")

    gt_valid: np.ndarray = np.isfinite(ground_truth) & valid_mask
    valid_count: int = int(np.sum(gt_valid))
    if valid_count == 0:
        raise ValueError("ground_truth 沒有可用的有效像素。")

    disparity_float: np.ndarray = disparity.astype(np.float32)
    gt_float: np.ndarray = ground_truth.astype(np.float32)
    diff: np.ndarray = np.abs(disparity_float - gt_float)
    diff_valid: np.ndarray = diff[gt_valid]
    bad_pixels: np.ndarray = diff_valid > bad_threshold
    pbm: float = float(np.mean(bad_pixels) * 100.0)
    rms: float = float(np.sqrt(np.mean(diff_valid ** 2)))
    return {
        "pbm": pbm,
        "rms": rms,
        "bad_threshold": float(bad_threshold),
        "valid_pixel_count": float(valid_count),
    }


def _write_metrics(path: Path, metrics: Dict[str, float]) -> None:
    """輸出評估指標到 JSON 檔案。"""
    with path.open("w", encoding="utf-8") as handler:
        json.dump(metrics, handler, ensure_ascii=True, indent=2, sort_keys=True)


def _write_json(path: Path, data: Dict[str, object]) -> None:
    """輸出 JSON 檔案。"""
    with path.open("w", encoding="utf-8") as handler:
        json.dump(data, handler, ensure_ascii=True, indent=2, sort_keys=True)


def _print_metrics(metrics: Dict[str, float]) -> None:
    """在終端輸出評估指標。"""
    pbm: float = metrics.get("pbm", float("nan"))
    rms: float = metrics.get("rms", float("nan"))
    bad_threshold: float = metrics.get("bad_threshold", float("nan"))
    valid_pixel_count: float = metrics.get("valid_pixel_count", float("nan"))
    message: str = (
        "評估結果:\n"
        f"  PBM: {pbm:.4f}%\n"
        f"  RMS: {rms:.4f}\n"
        f"  Bad Threshold: {bad_threshold:.4f}\n"
        f"  Valid Pixel Count: {valid_pixel_count:.0f}"
    )
    print(message)


def _validate_args(args: argparse.Namespace, dataset_root: Path, ndisp_map: Dict[str, int]) -> None:
    """驗證 CLI 參數。"""
    if args.all_datasets and args.dataset:
        raise ValueError("不可同時使用 --dataset 與 --all-datasets。")
    if args.all_datasets:
        if args.left or args.right or args.gt or args.gt_mask:
            raise ValueError("使用 --all-datasets 時不可提供 --left/--right/--gt/--gt-mask。")
        if args.dmax > 0:
            raise ValueError("使用 --all-datasets 時不可提供 --dmax。")
        if not dataset_root.exists():
            raise FileNotFoundError(f"找不到 dataset 根目錄: {dataset_root}")
        if not ndisp_map:
            raise ValueError("ndisp_summary.csv 無有效場景可用。")
        return
    if args.dataset:
        if args.left or args.right or args.gt or args.gt_mask:
            raise ValueError("使用 --dataset 時不可同時提供 --left/--right/--gt/--gt-mask。")
        if args.dmax < 0:
            raise ValueError("--dmax 必須為正整數。")
        if not dataset_root.exists():
            raise FileNotFoundError(f"找不到 dataset 根目錄: {dataset_root}")
        if args.dmax == 0 and args.dataset not in ndisp_map:
            raise ValueError(f"ndisp_summary.csv 缺少場景 {args.dataset} 的 dmax，請手動指定 --dmax。")
        return
    if not args.left or not args.right:
        raise ValueError("未使用 --dataset 時必須提供 --left 與 --right。")
    if args.dmax <= 0:
        raise ValueError("未使用 --dataset 時必須提供有效的 --dmax。")
    if not args.gt:
        raise ValueError("未使用 --dataset 時必須提供 --gt。")
    if not args.gt_mask:
        raise ValueError("未使用 --dataset 時必須提供 --gt-mask。")


def _run_scene(
    args: argparse.Namespace,
    run_dir: Path,
    left_path: Path,
    right_path: Path,
    gt_path: Path,
    gt_mask_path: Path,
    dmax: int,
    output_metrics: Optional[Path],
    dataset_name: str,
    all_datasets: bool,
    print_metrics: bool,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, float]:
    """執行單場景計算與評估。"""
    run_dir.mkdir(parents=True, exist_ok=True)
    output_gray: Path = run_dir / "disparity.png"
    output_color: Path = run_dir / "disparity_color.png"
    output_npz: Path = run_dir / "disparity.npz"
    left_img: np.ndarray = read_image(str(left_path))
    right_img: np.ndarray = read_image(str(right_path))
    left_gray: np.ndarray = to_gray(left_img)
    right_gray: np.ndarray = to_gray(right_img)
    disparity, min_cost = compute_disparity(
        left_gray,
        right_gray,
        dmax=dmax,
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
        show_progress=progress_callback is None,
        progress_callback=progress_callback,
    )
    _save_disparity_image(disparity, dmax, str(output_gray))
    _save_disparity_color_image(disparity, dmax, str(output_color))
    save_disparity_npz(str(output_npz), disparity, min_cost)
    gt_disp: np.ndarray = read_pfm(str(gt_path))
    gt_mask: np.ndarray = read_pgm_mask(str(gt_mask_path))
    metrics: Dict[str, float] = _compute_pbm_rms(disparity, gt_disp, gt_mask, args.bad_threshold)
    if output_metrics is not None:
        _write_metrics(output_metrics, metrics)
    if print_metrics:
        _print_metrics(metrics)
    metadata: Dict[str, str] = _build_run_metadata(
        args,
        run_dir,
        output_gray,
        output_color,
        output_npz,
        output_metrics,
        left_path,
        right_path,
        gt_path,
        gt_mask_path,
        dmax,
        dataset_name,
        all_datasets,
    )
    _write_run_metadata(run_dir / "params.json", metadata)
    _ = min_cost
    return metrics


def main() -> None:
    """簡易驗證入口：讀入左右影像並輸出視差圖。"""
    args: argparse.Namespace = _parse_args()
    dataset_root: Path = Path("dataset")
    ndisp_map: Dict[str, int] = {}
    if args.all_datasets or args.dataset:
        ndisp_map = _read_ndisp_summary(dataset_root / "ndisp_summary.csv")
    _validate_args(args, dataset_root, ndisp_map)
    timestamp: str = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir: Path = _create_run_directory("result", timestamp)
    if args.all_datasets:
        scenes: List[str] = sorted(
            scene for scene in ndisp_map.keys() if (dataset_root / scene).is_dir()
        )
        if not scenes:
            raise ValueError("dataset 內沒有可用場景。")
        per_scene_metrics: Dict[str, Dict[str, float]] = {}
        total_scenes: int = len(scenes)
        progress_state: DatasetProgressState = DatasetProgressState()
        for index, scene in enumerate(scenes, start=1):
            scene_dir: Path = run_dir / scene
            scene_dir.mkdir(parents=True, exist_ok=False)
            left_path, right_path, gt_path, gt_mask_path, dmax = _resolve_scene_inputs(
                dataset_root,
                scene,
                ndisp_map,
                0,
            )
            def progress_callback(current: int, total: int, label: str, idx: int = index) -> None:
                """更新 dataset 與步驟進度顯示。"""
                _print_dataset_stage_progress(
                    idx,
                    total_scenes,
                    current,
                    total,
                    label,
                    progress_state,
                )
            metrics: Dict[str, float] = _run_scene(
                args,
                scene_dir,
                left_path,
                right_path,
                gt_path,
                gt_mask_path,
                dmax,
                output_metrics=None,
                dataset_name=scene,
                all_datasets=True,
                print_metrics=False,
                progress_callback=progress_callback,
            )
            per_scene_metrics[scene] = metrics
        if progress_state.rendered:
            sys.stdout.write("\n")
            sys.stdout.flush()
        pbm_values: List[float] = [m["pbm"] for m in per_scene_metrics.values()]
        rms_values: List[float] = [m["rms"] for m in per_scene_metrics.values()]
        summary: Dict[str, object] = {
            "scene_count": len(per_scene_metrics),
            "bad_threshold": float(args.bad_threshold),
            "pbm_mean": float(np.mean(pbm_values)) if pbm_values else float("nan"),
            "rms_mean": float(np.mean(rms_values)) if rms_values else float("nan"),
            "scenes": per_scene_metrics,
        }
        _write_json(run_dir / "metrics_summary.json", summary)
        return
    if args.dataset:
        left_path, right_path, gt_path, gt_mask_path, dmax = _resolve_scene_inputs(
            dataset_root,
            args.dataset,
            ndisp_map,
            args.dmax,
        )
        _run_scene(
            args,
            run_dir,
            left_path,
            right_path,
            gt_path,
            gt_mask_path,
            dmax,
            output_metrics=run_dir / "metrics.json",
            dataset_name=args.dataset,
            all_datasets=False,
            print_metrics=True,
        )
        return
    _run_scene(
        args,
        run_dir,
        Path(args.left),
        Path(args.right),
        Path(args.gt),
        Path(args.gt_mask),
        args.dmax,
        output_metrics=run_dir / "metrics.json",
        dataset_name="",
        all_datasets=False,
        print_metrics=True,
    )


if __name__ == "__main__":
    main()
