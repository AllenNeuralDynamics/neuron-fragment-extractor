"""
Created on Sun July 16 14:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Miscellaneous helper routines.

"""

from google.cloud import storage
from pathlib import Path

import json
import os
import shutil


# --- GCS utils ---
def gcs_directory_exists(bucket_name, prefix):
    """
    Check whether a directory (prefix) exists in a GCS bucket.

    Parameters
    ----------
    bucket_name : str
        Name of the GCS bucket.
    prefix : str
        The "directory" path to check. Should end with '/'.

    Returns
    -------
    bool
        True if the directory contains any objects, False otherwise.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(client.list_blobs(bucket, prefix=prefix, max_results=1))
    return len(blobs) > 0


def list_gcs_filenames(gcs_dict, extension):
    """
    Lists all files in a GCS bucket with the given extension.

    Parameters
    ----------
    gcs_dict : dict
        ...
    extension : str
        File extension of filenames to be listed.

    Returns
    -------
    List[str]
        Filenames stored at "cloud" path with the given extension.
    """
    bucket = storage.Client().bucket(gcs_dict["bucket_name"])
    blobs = bucket.list_blobs(prefix=gcs_dict["path"])
    return [blob.name for blob in blobs if extension in blob.name]


def list_gcs_subdirectories(bucket_name, prefix):
    """
    Lists all direct subdirectories of a given prefix in a GCS bucket.

    Parameters
    ----------
    bucket : str
        Name of bucket to be read from.
    prefix : str
        Path to directory in "bucket".

    Returns
    -------
    List[str]
         Direct subdirectories of the given prefix.
    """
    # Load blobs
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(
        bucket_name, prefix=prefix, delimiter="/"
    )
    [blob.name for blob in blobs]

    # Parse directory contents
    prefix_depth = len(prefix.split("/"))
    subdirs = list()
    for prefix in blobs.prefixes:
        is_dir = prefix.endswith("/")
        is_direct_subdir = len(prefix.split("/")) - 1 == prefix_depth
        if is_dir and is_direct_subdir:
            subdirs.append(prefix)
    return subdirs


def read_json_from_gcs(bucket_name, blob_path):
    """
    Reads JSON file stored in a GCS bucket.

    Parameters
    ----------
    bucket_name : str
        Name of the GCS bucket containing the JSON file.
    blob_path : str
        Path to the JSON file within the GCS bucket.

    Returns
    -------
    dict
        Parsed JSON content as a Python dictionary.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    return json.loads(blob.download_as_text())


# --- Image utils ---
def init_bbox(origin, shape):
    """
    Gets the min and max coordinates of a bounding box based on "origin" and
    "shape".

    Parameters
    ----------
    origin : Tuple[int]
        Origin of bounding box which is assumed to be top-front-left corner.
    shape : Tuple[int]
        Shape of bounding box.

    Returns
    -------
    dict or None
        Bounding box.
    """
    if origin and shape:
        end = tuple([o + s for o, s in zip(origin, shape)])
        return {"min": origin, "max": end}
    else:
        return None


def is_contained(bbox, voxel):
    """
    Checks whether a given voxel is contained within the image bounding box
    specified by "bbox".

    Parameters
    ----------
    bbox : dict
        Dictionary with the keys "min" and "max" which specify a bounding box
        in an image.
    voxel : Tuple[int]
        Voxel coordinate to be checked.

    Returns
    -------
    bool
        Inidcation of whether "voxel" is contained within the given image
        bounding box.
    """
    above = any([v >= bbox_max for v, bbox_max in zip(voxel, bbox["max"])])
    below = any([v < bbox_min for v, bbox_min in zip(voxel, bbox["min"])])
    return False if above or below else True


def is_list_contained(bbox, voxels):
    """
    Checks whether a list of voxels is contained within a given image bounding
    box.

    Parameters
    ----------
    bbox : dict
        Dictionary with the keys "min" and "max" which specify a bounding box
        in an image.
    voxels : List[Tuple[int]]
        List of voxel coordinates to be checked.

    Returns
    -------
    bool
        Indication of whether every element in "voxels" is contained in
        "bbox".

    """
    return all([is_contained(bbox, voxel) for voxel in voxels])


def to_physical(voxel, anisotropy, shift=[0, 0, 0]):
    """
    Converts a voxel coordinate to a physical coordinate by applying the
    anisotropy scaling factors.

    Parameters
    ----------
    voxel : ArrayLike
        Voxel coordinate to be converted.
    anisotropy : ArrayLike
        Image to physical coordinates scaling factors to account for the
        anisotropy of the microscope.
    shift : ArrayLike, optional
        Shift to be applied to "voxel". The default is [0, 0, 0].

    Returns
    -------
    Tuple[float]
        Physical coordinate.
    """
    voxel = voxel[::-1]
    return tuple([voxel[i] * anisotropy[i] - shift[i] for i in range(3)])


def to_voxels(xyz, anisotropy):
    """
    Converts coordinate from a physical to voxel space.

    Parameters
    ----------
    xyz : ArrayLike
        Physical coordiante to be converted.
    anisotropy : Tuple[float]
        Image to physical coordinates scaling factors to account for the
        anisotropy of the microscope.

    Returns
    -------
    Tuple[int]
        Voxel coordinate.
    """
    voxel = [int(xyz[i] / anisotropy[i]) for i in range(3)]
    return tuple(voxel[::-1])


# --- IO utils ---
def read_json(path):
    """
    Reads JSON file stored at the given path.

    Parameters
    ----------
    path : str
        Path to JSON file to be read.

    Returns
    -------
    dict
        Contents of JSON file.
    """
    with open(path, "r") as f:
        return json.load(f)


def read_txt(path):
    """
    Reads txt file located at the given path.

    Parameters
    ----------
    path : str
        Path to txt file to be read.

    Returns
    -------
    str
        Contents of txt file.
    """
    with open(path, "r") as f:
        return f.read().splitlines()


def read_zip(zip_file, path):
    """
    Reads txt file located in a ZIP archive.

    Parameters
    ----------
    zip_file : ZipFile
        ZIP archive containing txt file to be read.
    path : str
        Path to txt file within ZIP archive to be read.

    Returns
    -------
    str
        Contents of text file in ZIP archive.
    """
    with zip_file.open(path) as f:
        return f.read().decode("utf-8")


# --- OS utils ---
def count_files(directory):
    """
    Counts the number of files in a given directory (non-recursively).

    Parameters
    ----------
    directory : str or Path
        Path to the directory.

    Returns
    -------
    int
        Number of files in the directory.
    """
    return sum(1 for f in Path(directory).iterdir() if f.is_file())


def listdir(path, extension=None):
    """
    Lists all files in the directory at "path". If an extension is
    provided, then only files containing "extension" are returned.

    Parameters
    ----------
    path : str
        Path to directory to be searched.
    extension : str, optional
       Extension of file type of interest. The default is None.

    Returns
    -------
    List[str]
        Filenames in directory with extension "extension" if provided.
        Otherwise, list of all files in directory.
    """
    if extension is None:
        return [f for f in os.listdir(path)]
    else:
        return [f for f in os.listdir(path) if f.endswith(extension)]


def list_paths(directory, extension=None):
    """
    Lists all paths within "directory" that end with "extension" if provided.

    Parameters
    ----------
    directory : str
        Directory to be searched.
    extension : str, optional
        If provided, only paths of files with the extension are returned. The
        default is None.

    Returns
    -------
    List[str]
        List of all paths within "directory".
    """
    paths = list()
    for f in listdir(directory, extension=extension):
        paths.append(os.path.join(directory, f))
    return paths


def mkdir(path, delete=False):
    """
    Creates a directory at "path".

    Parameters
    ----------
    path : str
        Path of directory to be created.
    delete : bool, optional
        Indication of whether to delete directory at path if it already
        exists. Default is False.

    Returns
    -------
    None
    """
    if delete:
        rmdir(path)
    if not os.path.exists(path):
        os.mkdir(path)


def rmdir(path):
    """
    Removes directory and all subdirectories located at "path".

    Parameters
    ----------
    path : str
        Path to directory and subdirectories to be deleted.

    Returns
    -------
    None
    """
    if os.path.exists(path):
        shutil.rmtree(path)
