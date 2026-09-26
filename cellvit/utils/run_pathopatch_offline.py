# -*- coding: utf-8 -*-

import argparse
import multiprocessing
import sys
from pathlib import Path

import numpy as np
import yaml
from openslide.lowlevel import OpenSlideError
from PIL import Image
from tqdm import tqdm

from pathopatch import logger
from pathopatch.cli import PreProcessingParser
import pathopatch.patch_extraction.patch_extraction as patch_extraction_module
from pathopatch.patch_extraction.patch_extraction import PreProcessor, queue_worker
from pathopatch.utils.patch_util import (
    calculate_background_ratio,
    get_intersected_labels,
    pad_tile,
    patch_to_tile_size,
)
from pathopatch.utils.tools import close_logger, end_timer, start_timer


def generate_lightweight_thumbnail(slide, slide_mpp, max_size=1024):
    """Return the minimum thumbnail set required by PathoPatch storage."""

    width, height = slide.dimensions
    longest_edge = max(width, height, 1)
    scale = min(1.0, max_size / float(longest_edge))
    thumbnail_width = max(1, int(width * scale))
    thumbnail_height = max(1, int(height * scale))
    thumbnail = Image.new("RGB", (thumbnail_width, thumbnail_height), color=(255, 255, 255))
    return {"thumbnail": thumbnail}


def load_wrapper_config(remaining_args) -> dict:
    config_path = None
    for index, arg in enumerate(remaining_args):
        if arg == "--config" and index + 1 < len(remaining_args):
            config_path = remaining_args[index + 1]
            break

    if config_path is None:
        return {}

    with Path(config_path).open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def load_thumbnail_max_size(cli_args: argparse.Namespace, wrapper_config: dict) -> int:
    if cli_args.thumbnail_max_size is not None:
        return cli_args.thumbnail_max_size

    return int(wrapper_config.get("thumbnail_max_size", 1024))


def should_skip_background_mask(cli_args: argparse.Namespace, wrapper_config: dict) -> bool:
    if cli_args.skip_background_mask:
        return True

    return bool(wrapper_config.get("skip_background_mask", False))


def generate_placeholder_masks() -> dict:
    placeholder = Image.new("RGB", (1, 1), color=(255, 255, 255))
    return {
        "mask_nogrid": placeholder.copy(),
        "mask": placeholder.copy(),
        "tissue_grid": placeholder.copy(),
    }


def get_slide_metadata_loader(slide_processor):
    metadata_loader = getattr(slide_processor, "slide_metadata_loader", None)
    if metadata_loader is not None:
        return metadata_loader

    metadata_loader = getattr(slide_processor, "image_metadata_loader", None)
    if metadata_loader is not None:
        return metadata_loader

    return slide_processor.image_loader


def compute_all_patch_coordinates(
    slide,
    tiles,
    target_level,
    target_patch_size,
    target_overlap,
    tissue_annotation_intersection_ratio,
    label_map,
    rescaling_factor=1.0,
    full_tile_size=2000,
    polygons=None,
    region_labels=None,
    tissue_annotation=None,
    mask_otsu=False,
    otsu_annotation="object",
    apply_prefilter=False,
    fast_mode=False,
):
    n_cols, n_rows = tiles.level_tiles[target_level]
    interesting_patches = [(row, col, 0.0) for row in range(n_rows) for col in range(n_cols)]
    return interesting_patches, generate_placeholder_masks(), {}


def resilient_process_queue(
    self,
    batch,
    wsi_file,
    wsi_metadata,
    level,
    polygons,
    region_labels,
    store,
):
    logger.debug(f"Started process {multiprocessing.current_process().name}")
    wsi_file = Path(wsi_file)
    wsi_file_global = wsi_file
    context_tiles = {}

    slide_metadata_loader = get_slide_metadata_loader(self)
    slide = slide_metadata_loader(str(wsi_file))
    slide_cu = self.image_loader(str(wsi_file))

    tile_size, overlap = patch_to_tile_size(
        self.config.patch_size, self.config.patch_overlap, self.rescaling_factor
    )

    tiles = self.deepzoomgenerator(
        meta_loader=slide,
        image_loader=slide_cu,
        tile_size=tile_size,
        overlap=overlap,
        limit_bounds=True,
    )

    if self.config.context_scales is not None:
        for c_scale in self.config.context_scales:
            overlap_context = int((c_scale - 1) * tile_size / 2) + overlap
            context_tiles[c_scale] = self.deepzoomgenerator(
                meta_loader=slide,
                image_loader=slide_cu,
                tile_size=tile_size,
                overlap=overlap_context,
                limit_bounds=True,
            )

    queue = multiprocessing.Queue()
    processes = []
    processed_count = multiprocessing.Value("i", 0)
    pbar = tqdm(total=len(batch), desc="Retrieving patches")

    for _ in range(self.config.processes):
        process = multiprocessing.Process(
            target=queue_worker,
            args=(
                queue,
                store,
                processed_count,
                self.config.normalize_stains,
                self.config.normalization_vector_json,
            ),
        )
        process.start()
        processes.append(process)

    patches_count = 0
    skipped_tiles = 0
    skipped_background = 0
    skipped_annotation = 0
    skipped_context = 0
    patch_result_list = []
    patch_distribution = {value: 0 for _, value in self.config.label_map.items()}

    start_time = start_timer()
    for row, col, _ in batch:
        pbar.update()
        patch_fname = f"{wsi_file_global.stem}_{row}_{col}.png"
        patch_yaml_name = f"{wsi_file_global.stem}_{row}_{col}.yaml"

        if self.config.context_scales is not None:
            context_patches = {scale: [] for scale in self.config.context_scales}
        else:
            context_patches = {}

        try:
            new_tile = np.array(tiles.get_tile(level, (col, row)), dtype=np.uint8)
        except OpenSlideError as error:
            skipped_tiles += 1
            logger.warning(
                f"Skipping unreadable tile in {wsi_file_global.name} at row={row}, col={col}: {error}"
            )
            continue

        patch = pad_tile(new_tile, tile_size + 2 * overlap, col, row)
        background_ratio = calculate_background_ratio(new_tile, self.config.patch_size)

        if background_ratio > 1 - self.config.min_intersection_ratio:
            skipped_background += 1
            intersected_labels = []
            ratio = {}
            patch_mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
            continue
        else:
            intersected_labels, ratio, patch_mask = get_intersected_labels(
                tile_size=tile_size,
                patch_overlap=self.config.patch_overlap,
                col=col,
                row=row,
                polygons=polygons,
                label_map=self.config.label_map,
                min_intersection_ratio=0,
                region_labels=region_labels,
                overlapping_labels=self.config.overlapping_labels,
                store_masks=self.config.store_masks,
            )
            if len(ratio) != 0:
                background_ratio = 1 - np.sum(ratio)
                ratio = {label: value for label, value in zip(intersected_labels, ratio)}
            if len(intersected_labels) == 0 and self.config.save_only_annotated_patches:
                skipped_annotation += 1
                continue

        patch_metadata = {
            "row": row,
            "col": col,
            "background_ratio": float(background_ratio),
            "intersected_labels": intersected_labels,
            "label_ratio": ratio,
            "wsi_metadata": wsi_metadata,
        }

        if not self.config.store_masks:
            patch_mask = None
        else:
            patch_metadata["mask"] = f"./masks/{Path(patch_fname).stem}_mask.npy"

        if self.config.context_scales is not None:
            patch_metadata["context_scales"] = []
            context_failed = False
            for c_scale in self.config.context_scales:
                try:
                    context_patch = np.array(
                        context_tiles[c_scale].get_tile(level, (col, row)),
                        dtype=np.uint8,
                    )
                except OpenSlideError as error:
                    skipped_tiles += 1
                    skipped_context += 1
                    context_failed = True
                    logger.warning(
                        f"Skipping unreadable context tile in {wsi_file_global.name} at row={row}, col={col}, scale={c_scale}: {error}"
                    )
                    break
                context_patch = pad_tile(context_patch, self.config.patch_size * c_scale, col, row)
                context_patch = np.array(
                    Image.fromarray(context_patch).resize(
                        (self.config.patch_size, self.config.patch_size)
                    ),
                    dtype=np.uint8,
                )
                context_patches[c_scale] = context_patch
                patch_metadata["context_scales"].append(c_scale)
            if context_failed:
                continue

        for patch_label in patch_metadata["intersected_labels"]:
            patch_distribution[patch_label] += 1

        patches_count += 1
        queue.put((patch, patch_metadata, patch_mask, context_patches, self.config.patch_size))

        patch_metadata.pop("wsi_metadata")
        patch_metadata["metadata_path"] = f"./metadata/{patch_yaml_name}"

        if self.save_context:
            patch_metadata["context_scales"] = {}
            for c_scale, _ in context_patches.items():
                context_name = f"{Path(patch_fname).stem}_context_{c_scale}.png"
                patch_metadata["context_scales"][c_scale] = f"./context/{context_name}"

        patch_result_list.append({patch_fname: patch_metadata})

    for _ in range(self.config.processes):
        queue.put(None)

    pbar.close()
    while not queue.empty():
        print(f"Progress: {processed_count.value}/{len(batch)}", end="\r")
        print("", end="", flush=True)

    for process in processes:
        process.join()
        process.close()

    logger.info(
        "Tile summary for %s: total=%d saved=%d unreadable=%d background_rejected=%d annotation_rejected=%d context_unreadable=%d",
        wsi_file_global.name,
        len(batch),
        patches_count,
        skipped_tiles,
        skipped_background,
        skipped_annotation,
        skipped_context,
    )
    logger.info(f"Skipped unreadable tiles: {skipped_tiles}")
    logger.info("Finished Processing and Storing. Took:")
    end_timer(start_time)
    return patches_count, patch_distribution, patch_result_list


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--thumbnail_max_size", type=int)
    parser.add_argument("--skip_background_mask", action="store_true")
    parser.add_argument("--keep_pathopatch_thumbnails", action="store_true")
    known_args, remaining_args = parser.parse_known_args()
    wrapper_config = load_wrapper_config(remaining_args)
    thumbnail_max_size = load_thumbnail_max_size(known_args, wrapper_config)
    skip_background_mask = should_skip_background_mask(known_args, wrapper_config)

    if not known_args.keep_pathopatch_thumbnails:
        patch_extraction_module.generate_thumbnails = (
            lambda slide, slide_mpp, sample_factors=None, mpp_factors=None: generate_lightweight_thumbnail(
                slide, slide_mpp, max_size=thumbnail_max_size
            )
        )
        print(f"PathoPatch wrapper: using placeholder thumbnails (max_size={thumbnail_max_size})")

    if skip_background_mask:
        patch_extraction_module.compute_interesting_patches = compute_all_patch_coordinates
        print("PathoPatch wrapper: skipping full-slide background mask generation")

    PreProcessor.process_queue = resilient_process_queue
    print("PathoPatch wrapper: skipping unreadable OpenSlide tiles during extraction")

    configuration_parser = PreProcessingParser()
    sys.argv = [sys.argv[0], *remaining_args]
    configuration, logger = configuration_parser.get_config()
    configuration_parser.store_config()

    slide_processor = PreProcessor(slide_processor_config=configuration)
    slide_processor.sample_patches_dataset()

    logger.info("Finished Preprocessing.")
    close_logger(logger)


if __name__ == "__main__":
    main()