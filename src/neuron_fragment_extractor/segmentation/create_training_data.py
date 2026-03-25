"""
Created on Tue Mar 16 11:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for extracting regions of ground truth skeletons that intersect with
image carve-outs

"""

import gcsfs
import json
import numpy as np
import os

from neuron_fragment_extractor.utils import img_util, util


def extract_volumes():
    # Open whole brain image
    img_path = os.path.join(
        img_util.find_img_path(bucket_name, "from_aind/", brain_id),
        str(0),
        )
    img = img_util.TensorStoreImage(img_path=img_path)

    # Extract volumes
    for i, physical_center in enumerate(physical_centers):
        # Create and store the array
        output_path = os.path.join(dataset_path, f"block_00{i}", "input.zarr")
        spec = get_tensorstore_spec(output_path, level=0)
        output_img = img_util.TensorStoreImage(spec=spec)
        img_util.write_zattrs(bucket_name, output_path, num_levels)
        print("Output Path:", output_path)

        # Read+Write image
        center = img_util.to_voxels(physical_center, anisotropy)
        read_sl = img_util.get_center_slices(center, img_shape[2:])
        write_sl = img_util.get_slices((0, 0, 0), img_shape[2:])
        output_img.write(img.read(read_sl), write_sl)


def get_tensorstore_spec(root_path, chunks=(128, 128, 128), level=0):
    """
    Creates a TensorStore specification for creating an image at the
    given path.

    Parameters
    ----------
    root_path : str
        Path to root directory of an OME-Zarr image.
    level : int, optional
        Image level to be written. Default is 0.

    Returns
    -------
    spec : dict
        TensorStore specification for creating the image.
    """
    # Extract info
    bucket_name, prefix = util.parse_cloud_path(root_path)
    shape = (1, 1, *(s // 2**level for s in img_shape[2:]))
    chunks = (1, 1, *(max(s // 2**level, 32) for s in chunks))

    # Create spec
    spec = {
        "driver": img_util.get_driver(root_path),
        "kvstore": {
            "driver": img_util.get_storage_driver(root_path),
            "bucket": bucket_name,
            "path": os.path.join(prefix, str(level)),
        },
        "context": {
            "cache_pool": {"total_bytes_limit": 1000000000},
            "cache_pool#remote": {"total_bytes_limit": 1000000000},
            "data_copy_concurrency": {"limit": 8},
        },
        "recheck_cached_metadata": False,
        "recheck_cached_data": False,
        "metadata": {
            "shape": shape,
            "zarr_format": 2,
            "fill_value": 0,
            "chunks": chunks,
            "compressor": {
                "id": "blosc",
                "cname": "zstd",
                "clevel": 3,
                "shuffle": 1,
            },
            "dimension_separator": "/",
            "dtype": "<u2",
        },
        "create": True,
        "delete_existing": True,
    }
    return spec


if __name__ == "__main__":
    # Parameters
    brain_id = "802449"
    bucket_name = "allen-nd-goog"
    physical_centers = [
        (35395.51, 20320.686, 6932.179),
        (35395, 20320, 6932),
        (33995, 22500, 6340),
        (29498, 12019, 6900),
        (26435, 18797, 6411),
        (35824, 20166, 6991),
        (36723, 22147, 11006),
        (37788, 21713, 10632),
        (36915, 20193, 8599),
        (37406, 20248, 8707),
    ]

    anisotropy = (0.748, 0.748, 1.0)
    img_shape = (1, 1, 512, 512, 512)
    num_levels = 3

    # Paths
    dataset_path = f"gs://allen-nd-goog/from_aind/crossover_blocks/{brain_id}"

    # Run code
    extract_volumes()