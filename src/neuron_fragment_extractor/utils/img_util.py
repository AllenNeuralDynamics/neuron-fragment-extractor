"""
Created on Fri Feb 20 14:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for working with images.

"""

import gcsfs
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorstore as ts
import zarr

from neuron_fragment_extractor.utils import util


# --- IO Utils ---
class TensorStoreImage:
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


def init_omezarr_image(
    img_path,
    img_shape,
    chunks=(1, 1, 64, 128, 128),
    voxel_size=(748, 748, 1000),
):
    """
    Creates an OME-Zarr image within a given GCS prefix.

    Parameters
    ----------
    img_path : str
        Path to GCS prefix that image is to be created.
    img_shape : Tuple[int]
        Shape of image to be created.
    chunks : Tuple[int], optional
        Shape of image shard. Default is (1, 1, 64, 128, 128).
    voxel_size : Tuple[int], optional
        Physical size (in nanometers) of voxels. Default is (748, 748, 1000).
    """
    # Initializations
    print(f"Creating OME-Zarr at {img_path} with shape {img_shape}")
    fs = gcsfs.GCSFileSystem()
    root_store = zarr.storage.FSStore(img_path, fs=fs)
    root_group = zarr.group(store=root_store, overwrite=True)

    # Create level 0
    store0 = zarr.storage.FSStore(os.path.join(img_path, str(0)), fs=fs)
    zarr.zeros(
        shape=img_shape,
        chunks=chunks,
        dtype="uint16",
        store=store0,
        overwrite=True
    )

    # Write metadata
    coord_transform = [
        {"type": "scale", "scale": [1, 1, *reversed(voxel_size)]}
    ]
    multiscales = [{
        "version": "0.4",
        "datasets": [{"path": "0"}],
        "axes": [
            {"name": "t", "type": "time"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space"},
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"}
        ],
        "coordinateTransformations": coord_transform
    }]
    multiscales_dict = {"multiscales": multiscales}
    root_store[".zattrs"] = json.dumps(multiscales_dict).encode("utf-8")


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


def plot_mips(img, output_path=None, vmax=None):
    """
    Plots the Maximum Intensity Projections (MIPs) of a 3D image along the XY,
    XZ, and YZ axes.

    Parameters
    ----------
    img : numpy.ndarray
        Input image to generate MIPs from.
    output_path : None or str, optional
        Path that plot is saved to if provided. Default is None.
    vmax : None or float, optional
        Brightness intensity used as upper limit of the colormap. Default is
        None.
    """
    vmax = vmax or np.percentile(img, 99.9)
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs_names = ["XY", "XZ", "YZ"]
    for i in range(3):
        if len(img.shape) == 5:
            mip = np.max(img[0, 0, ...], axis=i)
        else:
            mip = np.max(img, axis=i)

        axs[i].imshow(mip, vmax=vmax)
        axs[i].set_title(axs_names[i], fontsize=16)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
