#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator
from openslide.lowlevel import OpenSlideError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe MRXS/WSI readability by sampling DeepZoom tiles."
    )
    parser.add_argument("--wsi-folder", required=True, type=Path)
    parser.add_argument("--wsi-extension", default="mrxs")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument(
        "--downsample",
        type=int,
        default=2,
        help="Power-of-two downsample level to probe from level 0.",
    )
    parser.add_argument(
        "--max-slides",
        type=int,
        default=0,
        help="Limit how many slides to probe. 0 means all slides.",
    )
    return parser.parse_args()


def get_probe_level(tiles: DeepZoomGenerator, downsample: int) -> int:
    return max(0, tiles.level_count - downsample.bit_length())


def inspect_slide(wsi_path: Path, tile_size: int, overlap: int, downsample: int) -> dict:
    slide = OpenSlide(str(wsi_path))
    tiles = DeepZoomGenerator(slide, tile_size=tile_size, overlap=overlap, limit_bounds=True)
    level = get_probe_level(tiles, downsample)
    n_cols, n_rows = tiles.level_tiles[level]

    unreadable = 0
    first_error = None

    for row in range(n_rows):
        for col in range(n_cols):
            try:
                tile = tiles.get_tile(level, (col, row))
                tile.load()
            except OpenSlideError as error:
                unreadable += 1
                if first_error is None:
                    first_error = f"row={row}, col={col}: {error}"

    return {
        "slide": wsi_path.name,
        "rows": n_rows,
        "cols": n_cols,
        "total_tiles": n_rows * n_cols,
        "unreadable_tiles": unreadable,
        "first_error": first_error,
    }


def main() -> None:
    args = parse_args()
    print(f"Scanning for .{args.wsi_extension} slides in {args.wsi_folder}", flush=True)
    wsi_paths = sorted(args.wsi_folder.glob(f"**/*.{args.wsi_extension}"))
    if args.max_slides > 0:
        wsi_paths = wsi_paths[: args.max_slides]

    print(f"Found {len(wsi_paths)} slides in {args.wsi_folder}", flush=True)
    for index, wsi_path in enumerate(wsi_paths, start=1):
        print(f"[{index}/{len(wsi_paths)}] Checking {wsi_path.name}", flush=True)
        result = inspect_slide(
            wsi_path=wsi_path,
            tile_size=args.tile_size,
            overlap=args.overlap,
            downsample=args.downsample,
        )
        print(
            "  total_tiles={total_tiles} unreadable_tiles={unreadable_tiles}".format(
                **result
            ),
            flush=True,
        )
        if result["first_error"] is not None:
            print(f"  first_error={result['first_error']}", flush=True)


if __name__ == "__main__":
    main()