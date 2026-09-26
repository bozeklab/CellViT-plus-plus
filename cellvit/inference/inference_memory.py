# -*- coding: utf-8 -*-
# CellViT Inference Method for Patch-Wise Inference on a patches test set/Whole WSI
#
# Detect Cells with our Networks
# Patches dataset needs to have the follwoing requirements:
# Patch-Size must be 1024, with overlap of 64
#
# We provide preprocessing code here: ./preprocessing/patch_extraction/main_extraction.py
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

from pathlib import Path
from typing import Union

import pandas as pd
import pathopatch.patch_extraction.dataset as pathopatch_dataset_module
import ray
import torch
import tqdm
import ujson
import yaml
from openslide import OpenSlide
from cellvit.data.dataclass.cell_graph import CellGraphDataWSI
from cellvit.data.dataclass.wsi import WSIMetadata
from cellvit.inference.inference_disk import CellViTInference
from cellvit.inference.postprocessing_cupy import (
    BatchPoolingActor,
    DetectionCellPostProcessorCupy,
)
from pathopatch.patch_extraction.dataset import (
    LivePatchWSIDataloader,
    LivePatchWSIDataset,
    LivePatchWSIConfig,
)
import snappy
from cellvit.inference.wsi_meta import load_wsi_meta


def load_preprocessing_config(preprocessing_config: Union[Path, str, None]) -> dict:
    """Load optional PathoPatch preprocessing overrides from YAML."""

    if preprocessing_config is None:
        return {}

    config_path = Path(preprocessing_config)
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, dict):
        raise TypeError("Preprocessing config must contain a YAML mapping")
    return config


def generate_placeholder_masks() -> dict:
    return {"mask_nogrid": None, "mask": None, "tissue_grid": None}


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


def should_skip_background_mask(preprocessing_overrides: dict) -> bool:
    return bool(preprocessing_overrides.pop("skip_background_mask", False))


class CellViTInferenceMemory(CellViTInference):
    def __init__(
        self,
        model_path: Union[Path, str],
        gpu: int,
        outdir: Union[Path, str],
        classifier_path: Union[Path, str] = None,
        binary: bool = False,
        batch_size: int = 8,
        patch_size: int = 1024,
        overlap: int = 64,
        geojson: bool = False,
        graph: bool = False,
        compression: bool = False,
        enforce_mixed_precision: bool = False,
    ) -> None:
        super(CellViTInferenceMemory, self).__init__(
            model_path=model_path,
            classifier_path=classifier_path,
            binary=binary,
            gpu=gpu,
            batch_size=batch_size,
            patch_size=patch_size,
            overlap=overlap,
            geojson=geojson,
            graph=graph,
            compression=compression,
            enforce_mixed_precision=enforce_mixed_precision,
        )
        self.outdir = Path(outdir)

    def process_wsi(
        self,
        wsi_path: Union[Path, str],
        wsi_properties: dict = {},
        resolution: float = 0.25,
        preprocessing_config: Union[Path, str, None] = None,
        apply_prefilter: bool = False,
        filter_patches: bool = False,
        **kwargs,
    ) -> None:
        """Process a whole slide image with CellViT.

        Args:
            wsi_path (Union[Path, str]): Path to the whole slide image.
            wsi_properties (dict, optional): Optional WSI properties,
                Allowed keys are 'slide_mpp' and 'magnification'. Defaults to {}.
            resolution (float, optional): Target resolution. Defaults to 0.25.
            preprocessing_config (Union[Path, str, None], optional): Optional YAML
                file with PathoPatch preprocessing overrides.
            apply_prefilter (bool, optional): Prefilter. Defaults to False.
            filter_patches (bool, optional): Filter patches after processing. Defaults to False.
        """
        assert resolution in [0.25, 0.5], "Resolution must be one of [0.25, 0.5]"
        self.logger.info(f"Processing WSI: {wsi_path.name}")
        self.logger.info(f"Preparing WSI - Loading tissue region and prepare patches")
        slide_meta, target_mpp = load_wsi_meta(
            wsi_path=wsi_path,
            wsi_properties=wsi_properties,
            resolution=resolution,
            logger=self.logger,
        )

        preprocessing_overrides = load_preprocessing_config(preprocessing_config)
        skip_background_mask = should_skip_background_mask(preprocessing_overrides)
        controlled_fields = {
            "wsi_path",
            "wsi_properties",
            "patch_size",
            "patch_overlap",
            "target_mpp",
            "target_mpp_tolerance",
            "apply_prefilter",
            "filter_patches",
        }
        preprocessing_overrides = {
            key: value
            for key, value in preprocessing_overrides.items()
            if key not in controlled_fields
        }

        # setup wsi dataloader and postprocessor
        dataset_config = LivePatchWSIConfig(
            wsi_path=str(wsi_path),
            wsi_properties=wsi_properties,
            patch_size=self.patch_size,
            patch_overlap=(self.overlap / self.patch_size) * 100,
            target_mpp=target_mpp,
            apply_prefilter=apply_prefilter,
            filter_patches=filter_patches,
            target_mpp_tolerance=0.035,
            **preprocessing_overrides,
            **kwargs,
        )
        wsi_path = Path(wsi_path)

        original_compute_interesting_patches = None
        if skip_background_mask:
            self.logger.warning(
                "Skipping PathoPatch full-slide background mask generation for live inference."
            )
            original_compute_interesting_patches = (
                pathopatch_dataset_module.compute_interesting_patches
            )
            pathopatch_dataset_module.compute_interesting_patches = compute_all_patch_coordinates

        try:
            wsi_inference_dataset = LivePatchWSIDataset(
                slide_processor_config=dataset_config,
                logger=self.logger,
                transforms=self.inference_transforms,
            )
        finally:
            if original_compute_interesting_patches is not None:
                pathopatch_dataset_module.compute_interesting_patches = (
                    original_compute_interesting_patches
                )
        wsi_inference_dataloader = LivePatchWSIDataloader(
            dataset=wsi_inference_dataset, batch_size=self.batch_size, shuffle=False
        )
        wsi = WSIMetadata(
            name=wsi_path.name,
            slide_path=wsi_path,
            metadata=wsi_inference_dataset.wsi_metadata,
        )

        self.outdir.mkdir(exist_ok=True, parents=True)

        # global postprocessor
        postprocessor = DetectionCellPostProcessorCupy(
            wsi=wsi,
            nr_types=self.run_conf["data"]["num_nuclei_classes"],
            resolution=resolution,
            classifier=self.classifier,
            binary=self.binary,
        )

        # create ray actors for batch-wise postprocessing
        self._initialize_ray()
        batch_pooling_actors = [
            BatchPoolingActor.remote(postprocessor, self.run_conf)
            for i in range(self.ray_actors)
        ]

        call_ids = []
        inference_results = []

        self.logger.info("Extracting cells using CellViT...")
        with torch.no_grad():
            total_batches = len(wsi_inference_dataloader)
            batch_iterator = iter(wsi_inference_dataloader)
            pbar = tqdm.tqdm(total=total_batches)
            batch_num = 0
            while True:
                try:
                    batch = next(batch_iterator)
                except StopIteration:
                    break
                except IndexError:
                    self.logger.warning(
                        "PathoPatcher returned an empty batch during WSI iteration; skipping iterator step."
                    )
                    continue
                if not isinstance(batch[0], torch.Tensor) or batch[0].numel() == 0:
                    continue
                patches = batch[0].to(self.device)
                metadata = batch[1]
                batch_actor = batch_pooling_actors[batch_num % self.ray_actors]

                if self.mixed_precision:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        predictions = self.model.forward(patches, retrieve_tokens=True)
                else:
                    predictions = self.model.forward(patches, retrieve_tokens=True)
                predictions = self.apply_softmax_reorder(predictions)
                call_id = batch_actor.convert_batch_to_graph_nodes.remote(
                    predictions, metadata
                )
                call_ids.append(call_id)
                call_ids = self._drain_ray_tasks(call_ids, inference_results)
                pbar.update(1)
                batch_num += 1

            self.logger.info("Waiting for final batches to be processed...")
            self._drain_ray_tasks(call_ids, inference_results, drain_all=True)
        del pbar
        [ray.kill(batch_actor) for batch_actor in batch_pooling_actors]

        # unpack inference results
        cell_dict_wsi = []  # for storing all cell information
        cell_dict_detection = []  # for storing only the centroids

        graph_data = {
            "cell_tokens": [],
            "positions": [],
            "metadata": {
                "wsi_metadata": wsi.metadata,
                "nuclei_types": self.label_map,
            },
        }

        self.logger.info("Unpack Batches")
        for batch_results in inference_results:
            (
                batch_complete_dict,
                batch_detection,
                batch_cell_tokens,
                batch_cell_positions,
            ) = batch_results
            cell_dict_wsi = cell_dict_wsi + batch_complete_dict
            cell_dict_detection = cell_dict_detection + batch_detection
            graph_data["cell_tokens"] = graph_data["cell_tokens"] + batch_cell_tokens
            graph_data["positions"] = graph_data["positions"] + batch_cell_positions

        # cleaning overlapping cells
        if len(cell_dict_wsi) == 0:
            self.logger.warning("No cells have been extracted")
            return
        keep_idx = self._post_process_edge_cells(cell_list=cell_dict_wsi)
        cell_dict_wsi = [cell_dict_wsi[idx_c] for idx_c in keep_idx]
        cell_dict_detection = [cell_dict_detection[idx_c] for idx_c in keep_idx]
        graph_data["cell_tokens"] = [
            graph_data["cell_tokens"][idx_c] for idx_c in keep_idx
        ]
        graph_data["positions"] = [graph_data["positions"][idx_c] for idx_c in keep_idx]
        self.logger.info(f"Detected cells after cleaning: {len(keep_idx)}")

        # reallign grid if interpolation was used (including target_mpp_tolerance)
        if (
            not wsi.metadata["base_mpp"] - 0.035
            <= wsi.metadata["target_patch_mpp"]
            <= wsi.metadata["base_mpp"] + 0.035
        ):
            cell_dict_wsi, cell_dict_detection = self._reallign_grid(
                cell_dict_wsi=cell_dict_wsi,
                cell_dict_detection=cell_dict_detection,
                rescaling_factor=wsi.metadata["target_patch_mpp"]
                / wsi.metadata["base_mpp"],
            )

        # Coordinates are level-0 pixels relative to the non-empty slide region (PathoPatch
        # tiles with limit_bounds=True). Store that region's offset so consumers can add it.
        slide_props = OpenSlide(str(wsi_path)).properties
        wsi.metadata["coordinate_frame"] = "level0_relative_to_bounds"
        wsi.metadata["level0_offset_xy"] = [
            int(slide_props.get("openslide.bounds-x", 0)),
            int(slide_props.get("openslide.bounds-y", 0)),
        ]

        # saving/storing
        output_wsi_name = wsi_path.name.split(".")[0]
        cell_dict_wsi = {
            "wsi_metadata": wsi.metadata,
            "type_map": self.label_map,
            "cells": cell_dict_wsi,
        }
        if self.compression:
            with open(
                str(self.outdir / f"{output_wsi_name}_cells.json.snappy"), "wb"
            ) as outfile:
                compressed_data = snappy.compress(ujson.dumps(cell_dict_wsi, outfile))
                outfile.write(compressed_data)
        else:
            with open(
                str(self.outdir / f"{output_wsi_name}_cells.json"), "w"
            ) as outfile:
                ujson.dump(cell_dict_wsi, outfile)

        if self.geojson:
            self.logger.info("Converting segmentation to geojson")
            geojson_list = self._convert_json_geojson(cell_dict_wsi["cells"], True)
            if self.compression:
                with open(
                    str(self.outdir / f"{output_wsi_name}_cells.geojson.snappy"), "wb"
                ) as outfile:
                    compressed_data = snappy.compress(
                        ujson.dumps(geojson_list, outfile)
                    )
                    outfile.write(compressed_data)
            else:
                with open(
                    str(str(self.outdir / f"{output_wsi_name}_cells.geojson")), "w"
                ) as outfile:
                    ujson.dump(geojson_list, outfile)

        cell_dict_detection = {
            "wsi_metadata": wsi.metadata,
            "type_map": self.label_map,
            "cells": cell_dict_detection,
        }
        if self.compression:
            with open(
                str(self.outdir / f"{output_wsi_name}_cell_detection.json.snappy"), "wb"
            ) as outfile:
                compressed_data = snappy.compress(
                    ujson.dumps(cell_dict_detection, outfile)
                )
                outfile.write(compressed_data)
        else:
            with open(
                str(self.outdir / f"{output_wsi_name}_cell_detection.json"), "w"
            ) as outfile:
                ujson.dump(cell_dict_detection, outfile)
        if self.geojson:
            self.logger.info("Converting detection to geojson")
            geojson_list = self._convert_json_geojson(
                cell_dict_detection["cells"], False
            )
            if self.compression:
                with open(
                    str(
                        self.outdir / f"{output_wsi_name}_cell_detection.geojson.snappy"
                    ),
                    "wb",
                ) as outfile:
                    compressed_data = snappy.compress(
                        ujson.dumps(geojson_list, outfile)
                    )
                    outfile.write(compressed_data)
            else:
                with open(
                    str(str(self.outdir / f"{output_wsi_name}_cell_detection.geojson")),
                    "w",
                ) as outfile:
                    ujson.dump(geojson_list, outfile)

        # store graph
        if self.graph:
            self.logger.info(
                f"Create cell graph with embeddings and save it under: {str(self.outdir / f'{output_wsi_name}_cells.pt')}"
            )
            graph = CellGraphDataWSI(
                x=torch.stack(graph_data["cell_tokens"]),
                positions=torch.stack(graph_data["positions"]),
                metadata=graph_data["metadata"],
            )
            torch.save(graph, str(self.outdir / f"{output_wsi_name}_cells.pt"))

        # final output message
        cell_stats_df = pd.DataFrame(cell_dict_wsi["cells"])
        cell_stats = dict(cell_stats_df.value_counts("type"))
        verbose_stats = {self.label_map[k]: v for k, v in cell_stats.items()}
        self.logger.info(f"Finished with cell detection for WSI {output_wsi_name}")
        self.logger.info("Stats:")
        self.logger.info(f"{verbose_stats}")
