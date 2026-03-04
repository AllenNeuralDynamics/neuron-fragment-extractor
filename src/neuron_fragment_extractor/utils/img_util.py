"""
Created on Fri Feb 20 14:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for working with images.

"""

import gcsfs
import matplotlib.pyplot as plt
import numpy as np
import s3fs
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
        self.path = img_path
        self._load_image()

    def _load_image(self):
        """
        Loads image using the TensorStore library.
        """
        bucket_name, path = util.parse_cloud_path(self.path)
        self.img = ts.open(
            {
                "driver": get_driver(self.path),
                "kvstore": {
                    "driver": get_storage_driver(self.path),
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

    def shape(self):
        """
        Gets the shape of image.

        Returns
        -------
        Tuple[int]
            Shape of image.
        """
        return self.img.shape


def create_zarr_group(bucket_name, path):
    if util.is_gcs_path(path):
        gcs = gcsfs.GCSFileSystem(project=bucket_name)
        store = gcsfs.GCSMap(root=path, gcs=gcs, check=False, create=True)
    elif util.is_s3_path(path):
        s3 = s3fs.S3FileSystem(anon=False)
        store = s3fs.S3Map(root=path, s3=s3, check=False)
    else:
        raise Exception("Invalid path!")
    return zarr.open_group(store=store)


# --- Miscellaneous ---
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
