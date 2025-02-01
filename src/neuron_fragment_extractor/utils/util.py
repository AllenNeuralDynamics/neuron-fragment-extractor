"""
Created on Sun July 16 14:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Miscellaneous helper routines.

"""

import json
import os
import shutil


# --- io utils ---
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


# --- img utils ---
def init_bbox(origin, shape):
    """
    Gets the min and max coordinates of a bounding box based on "origin" and
    "shape".

    Parameters
    ----------
    origin : Tuple[int]
        Origin of bounding box which is assumed to be top-front-left corner.
    shape : tuple[int]
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


# --- miscellaneous ---
def mkdir(path, delete=False):
    """
    Creates a directory at "path".

    Parameters
    ----------
    path : str
        Path of directory to be created.
    delete : bool, optional
        Indication of whether to delete directory at path if it already
        exists. The default is False.

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
