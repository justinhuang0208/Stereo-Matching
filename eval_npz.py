from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

from stereo_io import load_npz, read_pfm, read_pgm_mask


def _parse_args() -> argparse.Namespace:
    """解析 CLI 參數。"""
    parser = argparse.ArgumentParser(description="Evaluate disparity NPZ with custom bad threshold")
    parser.add_argument("--npz", required=True, type=str, help="輸入 NPZ 檔案路徑")
    parser.add_argument("--gt", required=True, type=str, help="GT PFM 檔案路徑")
    parser.add_argument("--gt_mask", required=True, type=str, help="GT 遮罩 PGM 檔案路徑")
    parser.add_argument("--key", type=str, default="disparity", help="NPZ 內 disparity 的 key")
    parser.add_argument("--bad_threshold", type=float, default=1.0, help="Bad pixel 閾值")
    parser.add_argument("--output_json", type=str, default="", help="輸出 metrics JSON 檔案路徑")
    return parser.parse_args()


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


def _write_metrics(path: Path, metrics: Dict[str, float]) -> None:
    """輸出評估指標到 JSON 檔案。"""
    with path.open("w", encoding="utf-8") as handler:
        json.dump(metrics, handler, ensure_ascii=True, indent=2, sort_keys=True)


def _load_disparity_from_npz(path: str, key: str) -> np.ndarray:
    """從 NPZ 讀取 disparity 陣列。"""
    data: Dict[str, np.ndarray] = load_npz(path)
    if key not in data:
        raise ValueError(f"NPZ 不包含 key: {key}")
    disparity: np.ndarray = data[key]
    if disparity.ndim != 2:
        raise ValueError("disparity 必須為 2D。")
    return disparity


def main() -> None:
    """以自訂 bad threshold 評估 NPZ disparity。"""
    args: argparse.Namespace = _parse_args()
    disparity: np.ndarray = _load_disparity_from_npz(args.npz, args.key)
    ground_truth: np.ndarray = read_pfm(args.gt)
    valid_mask: np.ndarray = read_pgm_mask(args.gt_mask)
    metrics: Dict[str, float] = _compute_pbm_rms(disparity, ground_truth, valid_mask, args.bad_threshold)
    _print_metrics(metrics)
    if args.output_json:
        _write_metrics(Path(args.output_json), metrics)


if __name__ == "__main__":
    main()
