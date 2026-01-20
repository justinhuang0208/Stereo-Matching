from __future__ import annotations

import argparse

from stereo_io import convert_npz_to_pfm, convert_pfm_to_npz


def _parse_args() -> argparse.Namespace:
    """解析轉換工具 CLI 參數。"""
    parser = argparse.ArgumentParser(description="NPZ / PFM 轉換工具")
    parser.add_argument("--input", required=True, type=str, help="輸入檔案路徑")
    parser.add_argument("--output", required=True, type=str, help="輸出檔案路徑")
    parser.add_argument(
        "--mode",
        required=True,
        type=str,
        choices=["npz2pfm", "pfm2npz"],
        help="轉換方向",
    )
    parser.add_argument("--key", type=str, default="disparity", help="NPZ 內的陣列 key")
    return parser.parse_args()


def main() -> None:
    """轉換工具入口。"""
    args: argparse.Namespace = _parse_args()
    if args.mode == "npz2pfm":
        convert_npz_to_pfm(args.input, args.output, key=args.key)
        return
    convert_pfm_to_npz(args.input, args.output, key=args.key)


if __name__ == "__main__":
    main()
