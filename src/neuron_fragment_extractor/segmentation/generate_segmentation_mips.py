"""
Created on Tue Mar 10 13:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for generating maximum intensity projections (MIPs) incrementally along a
dimenion of a whole-brain segmentation.

"""

from concurrent.futures import as_completed, ThreadPoolExecutor
from tqdm import tqdm

import fastremap
import imageio
import numpy as np
import os

from neuron_fragment_extractor.utils.img_util import TensorStoreImage
from neuron_fragment_extractor.utils import img_util, util


def main():
    # Initializations
    permutation = get_permutation()
    rng = np.random.default_rng(0)
    util.mkdir(output_dir, delete=True)

    # Load image
    assert projection_dim in np.arange(2, 5)
    img = TensorStoreImage(img_path=img_path)
    img.permute_axes(permutation)

    print("Image Shape:", img.shape())
    print("MIPs Shape:", img.shape()[2:4])
    print("# MIPs:", len(np.arange(0, img.shape()[-1], step_size)))

    # Generate MIPs
    color_mapping = rng.integers(0, 256, size=(10**9, 3), dtype=np.uint8)
    color_mapping[0] = 0
    label_mapping = {0: 0}
    for z_start in tqdm(np.arange(0, img.shape()[-1], step_size)):
        # Compute MIP
        mip = generate_mip(img, z_start)
        mip = reassign_labels(mip, label_mapping)

        # Save result
        if len(label_mapping) > 1:
            path = os.path.join(output_dir, f"mip_{z_start}.png")
            imageio.imwrite(path, color_mapping[mip])


def generate_mip(img, z_start):

    def submit_job():
        try:
            start = next(iterator)
            thread = executor.submit(worker, start)
            pending[thread] = start
        except StopIteration:
            pass

    def worker(start):
        slices = img_util.get_slices(start, shape)
        patch = img.read(slices)
        return np.max(patch, axis=-1)

    # Initializations
    shape = (chunk_size, chunk_size, step_size)
    total_jobs = count_jobs(img)
    iterator = generate_jobs(img.shape(), z_start)
    mip = np.zeros(img.shape()[2:4], dtype=int)
    pending = dict()

    # Main
    with ThreadPoolExecutor(max_workers=64) as executor:
        # Assign initial threads
        for num_jobs in range(num_workers):
            submit_job()

        # Process threads
        while pending:
            # Wait for a job to complete
            thread = as_completed(pending.keys(), timeout=None).__next__()
            x_start, y_start, _ = pending.pop(thread)
            mip_xy = thread.result()

            # Store results
            x_slice = slice(x_start, x_start + chunk_size)
            y_slice = slice(y_start, y_start + chunk_size)
            mip[(x_slice, y_slice)] = mip_xy

            # Check whether to submit new job
            if num_jobs < total_jobs:
                submit_job()
                num_jobs += 1
    return mip


def reassign_labels(mip, label_mapping):
    mip = remove_small_segments(mip, 64)
    for label in fastremap.unique(mip):
        if label not in label_mapping:
            label_mapping[label] = len(label_mapping)
    return fastremap.remap(mip, label_mapping)


# --- Helpers ---
def count_jobs(img):
    return np.prod([img.shape()[d] // chunk_size for d in [2, 3]])


def dimension_iterator(length, step_size):
    return range(0, length - step_size, step_size)


def generate_jobs(img_shape, z_start):
    for x_start in dimension_iterator(img_shape[2], step_size):
        for y_start in dimension_iterator(img_shape[3], step_size):
            yield (x_start, y_start, z_start)


def get_permutation():
    permutation = np.arange(5)
    permutation[projection_dim], permutation[4] = 4, projection_dim
    return permutation.tolist()


def remove_small_segments(label_mask, min_size):
    """
    Removes small segments from a label mask.

    Parameters
    ----------
    label_mask : numpy.ndarray
        Integer array representing a segmentation mask. Each unique
        nonzero value corresponds to a distinct segment.
    min_size : int
        Minimum size (in voxels) for a segment to be kept.

    Returns
    -------
    label_mask : numpy.ndarray
        A new label mask of the same shape as the input, with only
        the retained segments renumbered contiguously. Background
        voxels remain labeled as 0.
    """
    ids, cnts = fastremap.unique(label_mask, return_counts=True)
    ids = [i for i, cnt in zip(ids, cnts) if cnt > min_size and i != 0]
    return fastremap.mask_except(label_mask, ids)


if __name__ == "__main__":
    # Parameters
    chunk_size = 512
    projection_dim = 4
    num_workers = 32
    step_size = 512

    # Paths
    img_path = "gs://allen-nd-goog/from_google/784802/whole_brain/jin_masked_mean40_stddev105"
    output_dir = "/home/jupyter/results/mips_whole-brain/784802"

    # Run code
    main()
