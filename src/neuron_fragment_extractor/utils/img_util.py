"""
Created on Fri Feb 20 14:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for working with images.

"""

from numcodecs import Blosc
from ome_zarr.writer import write_multiscale
from xarray_multiscale import multiscale
from xarray_multiscale.reducers import windowed_mode

import numpy as np
import tensorstore as ts
import zarr

from neuron_fragment_extractor.utils import util


# --- IO Utils ---
class TensorStoreReader:
    """
    Class that reads an image with TensorStore library.
    """

    def __init__(self, img_path):
        """
        Instantiates a TensorStoreReader object.

        Parameters
        ----------
        img_path : str
            Path to image.
        """
        self.img_path = img_path
        self._load_image()

    def _load_image(self):
        """
        Loads image using the TensorStore library.
        """
        bucket_name, path = util.parse_cloud_path(self.img_path)
        self.img = ts.open(
            {
                "driver": get_driver(self.img_path),
                "kvstore": {
                    "driver": get_storage_driver(self.img_path),
                    "bucket": bucket_name,
                    "path": path,
                },
                "context": {
                    "cache_pool": {"total_bytes_limit": 1000000000},
                    "cache_pool#remote": {"total_bytes_limit": 1000000000},
                    "data_copy_concurrency": {"limit": 8},
                },
                "recheck_cached_data": "open",
            }
        ).result()

    def read(self, slices):
        """
        Reads the patch specified by the given image slices.

        Parameters
        ----------
        slices : Tuple[slice]
            Slice objects specifying the region to extract from the image.

        Returns
        -------
        numpy.ndarray
            Image patch.
        """
        return self.img[slices].read().result()

    def shape(self):
        """
        Gets the shape of image.

        Returns
        -------
        Tuple[int]
            Shape of image.
        """
        return self.img.shape


def write_ome_zarr(
    img,
    output_path,
    chunks=(1, 1, 64, 128, 128),
    compressor=Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE),
    n_levels=1,
    scale_factors=(1, 1, 2, 2, 2),
    voxel_size=(748, 748, 1000),
):
    # Ensure 5D image (T, C, Z, Y, X)
    while img.ndim < 5:
        img = img[np.newaxis, ...]

    # Generate multiscale pyramid
    pyramid = multiscale(img, windowed_mode, scale_factors=scale_factors)[:n_levels]
    pyramid = [level.data for level in pyramid]

    # Prepare Zarr store
    store = zarr.DirectoryStore(output_path, dimension_separator="/")
    zgroup = zarr.open(store=store, mode="w")

    # Voxel size scaling for each level
    base_scale = np.array([1, 1, *reversed(voxel_size)])
    scales = [base_scale[:2].tolist() + (base_scale[2:] * 2**i).tolist() for i in range(n_levels)]
    coord_transforms = [[{"type": "scale", "scale": s}] for s in scales]

    # Write to OME-Zarr
    write_multiscale(
        pyramid=pyramid,
        group=zgroup,
        chunks=chunks,
        axes=[
            {"name": "t", "type": "time", "unit": "millisecond"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ],
        coordinate_transformations=coord_transforms,
        storage_options={"compressor": compressor},
    )


# --- Miscellaneous ---
def get_driver(img_path):
    """
    Gets the storage driver needed to read the image.

    Parameters
    ----------
    img_path : str
        Path to image

    Returns
    -------
    str
        Storage driver needed to read the image.
    """
    if ".zarr" in img_path:
        return "zarr"
    elif ".n5" in img_path:
        return "n5"
    else:
        raise ValueError(f"Unsupported image format: {img_path}")


def get_slices(center, shape):
    """
    Gets the start and end indices of the chunk to be read.

    Parameters
    ----------
    center : Tuple[int]
        Center of image patch to be read.
    shape : Tuple[int]
        Shape of image patch to be read.

    Return
    ------
    Tuple[slice]
        Slice objects used to index into the image.
    """
    start = [int(c - d // 2) for c, d in zip(center, shape)]
    return tuple(slice(s, s + d) for s, d in zip(start, shape))


def get_storage_driver(img_path):
    """
    Gets the storage driver needed to read the image.

    Parameters
    ----------
    img_path : str
        Image path to be checked.

    Returns
    -------
    str
        Storage driver needed to read the image.
    """
    if util.is_s3_path(img_path):
        return "s3"
    elif util.is_gcs_path(img_path):
        return "gcs"
    else:
        raise ValueError(f"Unsupported path type: {img_path}")


def is_patch_contained(center, patch_shape, image_shape):
    """
    Checks if the given image patch defined by "center" and "patch_shape" is
    contained in the image defined by "image_shape".

    Parameters
    ----------
    voxel : Tuple[int]
        Voxel coordinates to be checked.
    patch_shape : Tuple[int]
        Shape of patch.
    image_shape : Tuple[int], optional
        Shape of image containing the patch.

    Returns
    -------
    bool
        True if the patch is contained in the image.
    """
    # Convert to arrays
    center = np.asarray(center)
    patch_shape = np.asarray(patch_shape)
    image_shape = np.asarray(image_shape)

    # Compute patch vertices
    half = patch_shape // 2
    start = center - half
    end = start + patch_shape
    return np.all(start >= 0) and np.all(end <= image_shape)
