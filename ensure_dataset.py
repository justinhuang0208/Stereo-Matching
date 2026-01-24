#!/usr/bin/env python3
"""Ensure Middlebury 2014 perfect datasets are present and complete."""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DEFAULT_BASE_URL: str = "https://vision.middlebury.edu/stereo/data/scenes2014/zip"
DEFAULT_SCENES: List[str] = [
    "Adirondack",
    "Backpack",
    "Bicycle1",
    "Cable",
    "Classroom1",
    "Couch",
    "Flowers",
    "Jadeplant",
    "Mask",
    "Motorcycle",
    "Piano",
    "Pipes",
    "Playroom",
    "Playtable",
    "Recycle",
    "Shelves",
    "Shopvac",
    "Sticks",
    "Storage",
    "Sword1",
    "Sword2",
    "Umbrella",
    "Vintage",
]
DEFAULT_EXPECTED_FILES: List[str] = [
    "calib.txt",
    "im0.png",
    "im1.png",
    "im1E.png",
    "im1L.png",
    "disp0.pfm",
    "disp1.pfm",
    "disp0-sd.pfm",
    "disp1-sd.pfm",
    "disp0-n.pgm",
    "disp1-n.pgm",
]


def parse_csv_list(value: str) -> List[str]:
    """Parse a comma-separated list into items with surrounding whitespace removed."""
    items: List[str] = [item.strip() for item in value.split(",")]
    return [item for item in items if item]


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def scene_dir_name(scene: str) -> str:
    """Build the folder name for a given scene."""
    return f"{scene}-perfect"


def zip_name(scene: str) -> str:
    """Build the zip filename for a given scene."""
    return f"{scene}-perfect.zip"


def scene_dir(dataset_dir: Path, scene: str) -> Path:
    """Return the path to a scene directory."""
    return dataset_dir / scene_dir_name(scene)


def scene_zip_url(base_url: str, scene: str) -> str:
    """Return the download URL for a scene zip file."""
    return f"{base_url}/{zip_name(scene)}"


def missing_files_for_scene(scene_path: Path, expected_files: Sequence[str]) -> List[str]:
    """Return the list of missing or empty files for a scene directory."""
    missing: List[str] = []
    if not scene_path.is_dir():
        return ["__scene_dir__"]
    for filename in expected_files:
        candidate: Path = scene_path / filename
        if not candidate.is_file():
            missing.append(filename)
            continue
        if candidate.stat().st_size == 0:
            missing.append(filename)
    return missing


def dataset_completeness(
    dataset_dir: Path,
    scenes: Sequence[str],
    expected_files: Sequence[str],
) -> Tuple[bool, Dict[str, List[str]]]:
    """Check if all scenes contain the expected files."""
    missing_by_scene: Dict[str, List[str]] = {}
    for scene in scenes:
        missing: List[str] = missing_files_for_scene(scene_dir(dataset_dir, scene), expected_files)
        if missing:
            missing_by_scene[scene] = missing
    return (len(missing_by_scene) == 0, missing_by_scene)


def download_file(url: str, dest: Path, timeout: int) -> None:
    """Download a file from a URL to a destination path."""
    ensure_dir(dest.parent)
    tmp_path: Path = dest.with_suffix(dest.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with urllib.request.urlopen(url, timeout=timeout) as response:
        with tmp_path.open("wb") as output:
            shutil.copyfileobj(response, output)
    tmp_path.replace(dest)


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    """Extract a zip archive into the output directory."""
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)


def download_and_extract_all(
    dataset_dir: Path,
    scenes: Sequence[str],
    base_url: str,
    zips_dir: Path,
    timeout: int,
) -> None:
    """Download and extract all scene zip files."""
    ensure_dir(dataset_dir)
    ensure_dir(zips_dir)
    for scene in scenes:
        url: str = scene_zip_url(base_url, scene)
        zip_path: Path = zips_dir / zip_name(scene)
        print(f"Downloading {url}")
        download_file(url, zip_path, timeout)
        print(f"Extracting {zip_path}")
        extract_zip(zip_path, dataset_dir)


def format_missing_report(missing_by_scene: Dict[str, List[str]]) -> str:
    """Create a human-readable report of missing files."""
    lines: List[str] = []
    for scene, missing in sorted(missing_by_scene.items()):
        if missing == ["__scene_dir__"]:
            lines.append(f"- {scene}: missing directory")
            continue
        lines.append(f"- {scene}: missing {', '.join(missing)}")
    return "\n".join(lines)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ensure Middlebury 2014 perfect datasets are present and complete."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset"),
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help="Base URL hosting the scene zip files.",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        default="",
        help="Comma-separated scene names to download (default: built-in list).",
    )
    parser.add_argument(
        "--expected-files",
        type=str,
        default="",
        help="Comma-separated expected files in each scene directory.",
    )
    parser.add_argument(
        "--zips-dir",
        type=Path,
        default=None,
        help="Directory to store downloaded zip files (default: dataset/zips).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Download timeout in seconds.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check completeness; do not download.",
    )
    return parser.parse_args(argv)


def resolve_scenes(scenes_arg: str, default_scenes: Sequence[str]) -> List[str]:
    """Resolve the scene list from the CLI argument or defaults."""
    if scenes_arg.strip():
        return parse_csv_list(scenes_arg)
    return list(default_scenes)


def resolve_expected_files(expected_files_arg: str, default_files: Sequence[str]) -> List[str]:
    """Resolve the expected file list from the CLI argument or defaults."""
    if expected_files_arg.strip():
        return parse_csv_list(expected_files_arg)
    return list(default_files)


def determine_zips_dir(dataset_dir: Path, zips_dir: Path | None) -> Path:
    """Resolve the zip storage directory."""
    if zips_dir is not None:
        return zips_dir
    return dataset_dir / "zips"


def main(argv: Sequence[str]) -> int:
    """Run the dataset completeness check and optional download."""
    args: argparse.Namespace = parse_args(argv)
    dataset_dir: Path = args.dataset_dir
    scenes: List[str] = resolve_scenes(args.scenes, DEFAULT_SCENES)
    expected_files: List[str] = resolve_expected_files(args.expected_files, DEFAULT_EXPECTED_FILES)
    zips_dir: Path = determine_zips_dir(dataset_dir, args.zips_dir)

    is_complete, missing_by_scene = dataset_completeness(dataset_dir, scenes, expected_files)
    if is_complete:
        print("Dataset is complete.")
        return 0

    print("Dataset is incomplete:")
    print(format_missing_report(missing_by_scene))

    if args.check_only:
        return 1

    print("Downloading all scenes to restore completeness...")
    download_and_extract_all(
        dataset_dir=dataset_dir,
        scenes=scenes,
        base_url=args.base_url,
        zips_dir=zips_dir,
        timeout=args.timeout,
    )
    is_complete_after, missing_after = dataset_completeness(dataset_dir, scenes, expected_files)
    if not is_complete_after:
        print("Download finished, but dataset is still incomplete:")
        print(format_missing_report(missing_after))
        return 2

    print("Dataset download complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
