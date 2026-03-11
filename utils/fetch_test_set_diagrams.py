"""Download SVGs from a JSON dataset, convert them to PNG, and store project-relative paths."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSON_PATH = ROOT / "data" / "test_set_450.json"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "diagrams"
DEFAULT_PATH_KEY = "file-path"
DEFAULT_RENDERER = "auto"
DEFAULT_PADDING_PX = 28
DEFAULT_ZOOM = 2.0

_LENGTH_PATTERN = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*([a-zA-Z%]*)\s*$")
_UNIT_TO_PX = {
    "": 1.0,
    "px": 1.0,
    "in": 96.0,
    "cm": 96.0 / 2.54,
    "mm": 96.0 / 25.4,
    "pt": 96.0 / 72.0,
    "pc": 16.0,
    "q": 96.0 / (25.4 * 4.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download SVGs referenced by a JSON array, convert them to PNG, and "
            "write a project-relative path back into each item."
        )
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=DEFAULT_JSON_PATH,
        help=f"Path to the JSON file to update. Default: {DEFAULT_JSON_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for PNG outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--path-key",
        default=DEFAULT_PATH_KEY,
        help=f"JSON key used for the relative PNG path. Default: {DEFAULT_PATH_KEY}",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds. Default: 30",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not re-download or re-render PNG files that already exist.",
    )
    parser.add_argument(
        "--renderer",
        choices=("auto", "rsvg", "magick"),
        default=DEFAULT_RENDERER,
        help="SVG renderer backend. Default: auto (prefer rsvg-convert).",
    )
    parser.add_argument(
        "--padding-px",
        type=float,
        default=DEFAULT_PADDING_PX,
        help=(
            "Extra white padding around each rendered SVG in output pixels. "
            f"Default: {DEFAULT_PADDING_PX}"
        ),
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=DEFAULT_ZOOM,
        help=f"Rasterization zoom factor. Default: {DEFAULT_ZOOM}",
    )
    return parser.parse_args()


def load_items(json_path: Path) -> list[dict[str, Any]]:
    with json_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array in {json_path}")

    items: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Expected object at index {index} in {json_path}")
        items.append(item)
    return items


def resolve_renderer(choice: str) -> str:
    if choice == "auto":
        if shutil.which("rsvg-convert"):
            return "rsvg"
        if shutil.which("magick"):
            return "magick"
        raise RuntimeError(
            "No supported SVG renderer found. Install 'rsvg-convert' or 'magick'."
        )

    if choice == "rsvg" and not shutil.which("rsvg-convert"):
        raise RuntimeError("Requested renderer 'rsvg' but 'rsvg-convert' was not found.")
    if choice == "magick" and not shutil.which("magick"):
        raise RuntimeError("Requested renderer 'magick' but 'magick' was not found.")
    return choice


def parse_svg_length_to_px(length_value: str | None) -> float | None:
    if not length_value:
        return None
    match = _LENGTH_PATTERN.match(length_value)
    if not match:
        return None
    number = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "%":
        return None
    scale = _UNIT_TO_PX.get(unit)
    if scale is None:
        return None
    return number * scale


def infer_svg_size_px(svg_bytes: bytes) -> tuple[float, float]:
    fallback = (1024.0, 1024.0)
    try:
        root = ET.fromstring(svg_bytes)
    except ET.ParseError:
        return fallback

    width_px = parse_svg_length_to_px(root.get("width"))
    height_px = parse_svg_length_to_px(root.get("height"))
    if width_px and height_px and width_px > 0 and height_px > 0:
        return (width_px, height_px)

    view_box = root.get("viewBox")
    if not view_box:
        return fallback

    parts = view_box.replace(",", " ").split()
    if len(parts) != 4:
        return fallback
    try:
        width = float(parts[2])
        height = float(parts[3])
    except ValueError:
        return fallback
    if width <= 0 or height <= 0:
        return fallback
    return (width, height)


def download_svg(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "image/svg+xml,image/*;q=0.8,*/*;q=0.5",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def convert_with_rsvg(
    svg_path: Path,
    png_path: Path,
    *,
    width_px: float,
    height_px: float,
    padding_px: float,
    zoom: float,
) -> None:
    left = max(0, int(round(padding_px * zoom)))
    top = max(0, int(round(padding_px * zoom)))
    page_width = max(1, int(round((width_px + (padding_px * 2)) * zoom)))
    page_height = max(1, int(round((height_px + (padding_px * 2)) * zoom)))
    subprocess.run(
        [
            "rsvg-convert",
            str(svg_path),
            "-f",
            "png",
            "-o",
            str(png_path),
            "--zoom",
            str(zoom),
            "--left",
            str(left),
            "--top",
            str(top),
            "--page-width",
            str(page_width),
            "--page-height",
            str(page_height),
            "--background-color",
            "white",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def convert_with_magick(svg_path: Path, png_path: Path, *, padding_px: float, zoom: float) -> None:
    density = max(96, int(round(96 * zoom)))
    border = max(0, int(round(padding_px)))
    subprocess.run(
        [
            "magick",
            "-density",
            str(density),
            str(svg_path),
            "-background",
            "white",
            "-alpha",
            "remove",
            "-alpha",
            "off",
            "-bordercolor",
            "white",
            "-border",
            f"{border}x{border}",
            str(png_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def convert_svg_bytes_to_png(
    svg_bytes: bytes,
    png_path: Path,
    *,
    renderer: str,
    padding_px: float,
    zoom: float,
) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        svg_path = Path(tmp_dir) / "source.svg"
        svg_path.write_bytes(svg_bytes)

        if renderer == "rsvg":
            width_px, height_px = infer_svg_size_px(svg_bytes)
            convert_with_rsvg(
                svg_path,
                png_path,
                width_px=width_px,
                height_px=height_px,
                padding_px=padding_px,
                zoom=zoom,
            )
            return

        convert_with_magick(svg_path, png_path, padding_px=padding_px, zoom=zoom)


def build_output_path(output_dir: Path, item: dict[str, Any], index: int) -> Path:
    idx_value = str(item.get("idx", index))
    stem = f"diagram_{idx_value}"
    candidate = output_dir / f"{stem}.png"
    suffix = 1
    while candidate.exists() and candidate.stem != stem:
        suffix += 1
        candidate = output_dir / f"{stem}_{suffix}.png"
    return candidate


def write_items(json_path: Path, items: list[dict[str, Any]]) -> None:
    json_path.write_text(
        json.dumps(items, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def process_dataset(
    json_path: Path,
    output_dir: Path,
    *,
    path_key: str,
    timeout: float,
    skip_existing: bool,
    renderer: str,
    padding_px: float,
    zoom: float,
) -> None:
    resolved_renderer = resolve_renderer(renderer)
    print(f"[info] using renderer: {resolved_renderer}")
    items = load_items(json_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for index, item in enumerate(items):
        url = item.get("url")
        if not isinstance(url, str) or not url.strip():
            failures.append(f"index {index}: missing url")
            continue

        png_path = build_output_path(output_dir, item, index)
        relative_path = png_path.relative_to(ROOT).as_posix()

        try:
            if not (skip_existing and png_path.exists()):
                svg_bytes = download_svg(url.strip(), timeout)
                convert_svg_bytes_to_png(
                    svg_bytes,
                    png_path,
                    renderer=resolved_renderer,
                    padding_px=padding_px,
                    zoom=zoom,
                )
            item[path_key] = relative_path
            print(f"[ok] {item.get('idx', index)} -> {relative_path}")
        except (
            OSError,
            urllib.error.URLError,
            urllib.error.HTTPError,
            subprocess.CalledProcessError,
        ) as exc:
            failures.append(f"index {index} ({item.get('idx', index)}): {exc}")
            print(f"[error] {item.get('idx', index)}: {exc}")

    write_items(json_path, items)

    if failures:
        raise RuntimeError(
            "Completed with failures:\n" + "\n".join(failures)
        )


def main() -> None:
    args = parse_args()
    json_path = args.json_path.resolve()
    output_dir = args.output_dir.resolve()
    process_dataset(
        json_path,
        output_dir,
        path_key=args.path_key,
        timeout=args.timeout,
        skip_existing=args.skip_existing,
        renderer=args.renderer,
        padding_px=args.padding_px,
        zoom=args.zoom,
    )


if __name__ == "__main__":
    main()
