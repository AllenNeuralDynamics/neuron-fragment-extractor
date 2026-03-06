"""
Created on Fri Feb 20 14:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for working with images.

"""

from scipy.ndimage import zoom

import matplotlib.pyplot as plt
import numpy as np
import tensorstore as ts

from neuron_fragment_extractor.utils import util


# --- IO Utils ---
class TensorStoreImage:
    """
    Class that reads an image with TensorStore library.
    """

    def __init__(self, img_path=None, spec=None):
        """
        Instantiates a TensorStoreReader object.

        Parameters
        ----------
        img_path : str
            Path to image.
        """
        assert img_path or spec
        self.spec = spec or self.create_spec(img_path)
        self.img = ts.open(self.spec).result()

    # --- Core Routines ---
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
        try:
            patch = self.img[slices].read().result()
        except ValueError:
            print(f"Error reading {slices} from img w/ shape {self.shape()}")
            patch_shape = tuple(s.stop - s.start for s in slices)
            patch = np.zeros(patch_shape)
        return patch

    def write(self, patch, slices):
        """
        Writes the given patch to the specified region.

        Parameters
        ----------
        patch : numpy.ndarray
            Image patch to be written.
        slices : Tuple[slice]
            Slice objects specifying the region to extract from the image.

        Returns
        -------
        numpy.ndarray
            Image patch.
        """
        self.img[slices] = patch

    # --- Helpers ---
    def create_spec(self, img_path):
        bucket_name, relative_path = util.parse_cloud_path(img_path)
        spec = {
            "driver": get_driver(img_path),
            "kvstore": {
                "driver": get_storage_driver(img_path),
                "bucket": bucket_name,
                "path": relative_path,
            },
            "context": {
                "cache_pool": {"total_bytes_limit": 1000000000},
                "cache_pool#remote": {"total_bytes_limit": 1000000000},
                "data_copy_concurrency": {"limit": 8},
            },
        }
        return spec

    def shape(self):
        """
        Gets the shape of image.

        Returns
        -------
        Tuple[int]
            Shape of image.
        """
        return self.img.shape


# --- Miscellaneous ---
def resize(img, new_shape):
    """
    Resize a 3D image to the specified new shape using linear interpolation.

    Parameters
    ----------
    img : numpy.ndarray
        Input 3D image array with shape (depth, height, width).
    new_shape : Tuple[int]
        Desired output shape as (new_depth, new_height, new_width).

    Returns
    -------
    numpy.ndarray
        Resized 3D image with shape equal to "new_shape".
    """
    zoom_factors = np.array(new_shape) / np.array(img.shape)
    return zoom(img, zoom_factors, order=0, prefilter=False)


def find_img_path(bucket_name, root_dir, brain_id):
    """
    Finds the path to a whole-brain dataset stored in a GCS bucket.

    Parameters:
    ----------
    bucket_name : str
        Name of the GCS bucket where the images are stored.
    root_dir : str
        Path to the directory in the GCS bucket where the image is expected to
        be located.
    dataset_name : str
        Name of the dataset to be searched for within the subdirectories.

    Returns:
    -------
    str
        Path of the found dataset subdirectory within the specified GCS bucket.
    """
    for subdir in util.list_gcs_subdirectories(bucket_name, root_dir):
        if brain_id in subdir:
            img_path = f"gs://{bucket_name}/{subdir}whole-brain/fused.zarr"
            return img_path
    raise f"Dataset not found in {bucket_name} - {root_dir}"


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


def get_center_slices(center, shape):
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
    slices = tuple(slice(s, s + d) for s, d in zip(start, shape))
    return (0, 0, *slices)


def get_slices(voxel, shape):
    """
    Gets the start and end indices of the chunk to be read.

    Parameters
    ----------
    voxel : Tuple[int]
        Start voxel of the slices.
    shape : Tuple[int]
        Shape of image patch to be read.

    Return
    ------
    Tuple[slice]
        Slice objects used to index into the image.
    """
    slices = tuple(slice(v, v + d) for v, d in zip(voxel, shape))
    return (0, 0, *slices)


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
