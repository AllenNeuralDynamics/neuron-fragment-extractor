"""
Created on Fri Feb 20 12:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for generating image carveouts based on ground-truth neuron tracings. A
carveout is initialized as an empty image in the coordinate space of a whole-
brain volume. A ground-truth neuron is then traversed, and the corresponding
voxel intensities from the original image are copied into the carveout,
producing a sparse image containing only the neuron's signal.

"""

from aind_data_transfer.transformations.ome_zarr import (
    downsample_and_store,
    write_ome_ngff_metadata
)
from google.cloud import storage
from tqdm import tqdm
from xarray_multiscale.reducers import windowed_max, windowed_mean

import asyncio
import dask.array as da
import networkx as nx
import numpy as np
import os
import queue
import threading

from neuron_fragment_extractor.graph_classes import SkeletonGraph
from neuron_fragment_extractor.utils.img_util import TensorStoreImage
from neuron_fragment_extractor.utils import img_util, util


def main():
    # Initializations
    gt_graph = load_skeletons()
    src_img = TensorStoreImage(input_img_path)
    img_shape = src_img.shape()

    # Generate carveouts
    pipeline = CarveOutPipeline(
        gt_graph, radial_shape, num_levels=num_levels, step_size=step_size
    )
    #pipeline.generate_raw("input.zarr", src_img)
    pipeline("mask.zarr", img_shape)

    # Write metadata
    metadata = ["SWC Names"] + gt_swc_names
    bucket, prefix = util.parse_cloud_path(output_gcs_dir)
    blob_name = os.path.join(prefix, "swc_names.txt")
    write_list_to_gcs(bucket, blob_name, metadata)


class CarveOutPipeline:

    def __init__(
        self,
        graph,
        radial_shape,
        block_shape=(1, 1, 128, 256, 256),
        chunks=(1, 1, 64, 128, 128),
        num_levels=1,
        num_readers=32,
        num_writers=1,
        prefetch=128,
        step_size=20,
        voxel_size=(1.0, 0.748, 0.748),
    ):
        """
        Instantiates a CarveOutPipeline object.

        Parameters
        ----------
        graph : SkeletonGraph
            Graph to be traversed to generate carve-out regions.
        radial_shape : Tuple[int]
            Shape of region centered about skeleton to be carved out.
        num_reader : int, optional
            Number of threads used to read image patches. Default is 16.
        num_writers : int, optional
            Number of threads used to write image patches. Default is 1.
        prefetch : int, optional
            Number of image patches to be prefeteched. Default is 128.
        step_size : float, optional
            Distance (in microns) between carved-out regions, measured along
            graph traversal. Default is 20.
        """
        # Instance attributes
        self.block_shape = block_shape
        self.chunks = chunks
        self.graph = graph
        self.num_levels = num_levels
        self.num_readers = num_readers
        self.num_writers = num_writers
        self.prefetch = prefetch
        self.radial_shape = radial_shape
        self.step_size = step_size
        self.voxel_size = voxel_size

    def __call__(self, filename, img_shape, src_img=None):
        # Create zarr group
        print("\nImage Name:", filename)
        img_path = os.path.join(output_gcs_dir, filename)
        bucket_name, prefix = util.parse_cloud_path(img_path)
        root_group = img_util.create_zarr_group(bucket_name, img_path)

        # Create and store the array
        print(f"Step 1: Create OME-Zarr at {img_path} with shape {img_shape}")
        root_group.create_dataset(
            "0",
            shape=img_shape,
            chunks=self.chunks,
            dtype="uint16",
            fill_value=None,
            overwrite=True
        )

        # Generate carve-out
        print("Step 2: Generate Image Carve-Out")
        dst_img = TensorStoreImage(os.path.join(img_path, str(0)))
        if filename == "mask.zarr":
            self.generate_mask(dst_img)
        else:
            self.generate_raw(src_img, dst_img)

        # Generate image pyramid
        print("Step 3: Generate Image Pyramid")
        arr = da.from_zarr(root_group["0"])
        reducer = windowed_max if filename == "mask.zarr" else windowed_mean
        downsample_and_store(
            arr=arr,
            group=root_group,
            n_lvls=self.num_levels,
            scale_factors=(1, 1, 2, 2, 2),
            block_shape=self.block_shape,
            reducer=reducer
        )

        # Write metadata
        print("Step 4: Write MetaData")
        write_ome_ngff_metadata(
            group=root_group,
            arr=arr,
            image_name=filename,
            n_lvls=self.num_levels,
            scale_factors=(2, 2, 2),
            voxel_size=self.voxel_size,
        )

        # Migrate result
        print("Step 5: Migrating from GCS to S3")
        self.migrate_result(filename)

    def generate_raw(self, src_img, dst_img):

        def traverse():
            for node in self.traverse_graph():
                if self.is_patch_contained(node, dst_img.shape()):
                    slices_q.put(self.node_to_slices(node))
            for _ in range(self.num_readers):
                slices_q.put(stop)

        def reader():
            while True:
                slices = slices_q.get()
                if slices is stop:
                    patch_q.put(stop)

                patch = src_img.read(slices)
                patch_q.put((slices, patch))

        def writer():
            finished_readers = 0
            while True:
                # Check whether to stop
                item = patch_q.get()
                if item is stop:
                    finished_readers += 1
                    if finished_readers == self.num_readers:
                        break
                    continue

                # Write patch
                slices, patch = item
                dst_img[slices] = patch
                pbar.update(1)

        # Initializations
        slices_q = queue.Queue(maxsize=self.prefetch)
        patch_q = queue.Queue(maxsize=self.prefetch)
        total_patches = self.count_patches(dst_img.shape())
        pbar = tqdm(total=total_patches, desc="Raw")
        stop = object()

        # Start threads
        threads = list()
        threads.append(threading.Thread(target=traverse, daemon=True))
        for _ in range(self.num_readers):
            threads.append(threading.Thread(target=reader, daemon=True))

        for _ in range(self.num_writers):
            threads.append(threading.Thread(target=writer, daemon=True))

        for t in threads:
            t.start()

        for t in threads:
            t.join()

    def generate_mask(self, dst_img):
        """
        Generates a binary mask that indicates which voxels are contained in
        the image carve-out.
        """

        def traverse():
            """
            Gets nodes to extract patches about by traversing the graph.
            """
            for node in self.traverse_graph():
                if self.is_patch_contained(node, dst_img.shape()):
                    slices_q.put(self.node_to_slices(node))

            for _ in range(self.num_readers):
                slices_q.put(stop)

        def writer():
            finished_readers = 0
            mask_patch = np.ones(self.radial_shape, dtype=np.uint16)
            while True:
                slices = slices_q.get()
                if slices is stop:
                    finished_readers += 1
                    if finished_readers == self.num_readers:
                        break
                    continue

                # Write patch
                dst_img.write(mask_patch, slices)
                pbar.update(1)

        # Initialize queues
        pbar = tqdm(total=self.count_patches(dst_img.shape()), desc="Mask")
        slices_q = queue.Queue(maxsize=self.prefetch)
        stop = object()

        # Start threads
        threads = [threading.Thread(target=traverse, daemon=True)]
        for _ in range(self.num_writers):
            threads.append(threading.Thread(target=writer, daemon=True))

        for t in threads:
            t.start()

        for t in threads:
            t.join()

    # --- Helpers ---
    def count_patches(self, img_shape):
        cnt = 0
        for node in self.traverse_graph():
            if self.is_patch_contained(node, img_shape):
                cnt += 1
        return cnt

    def is_patch_contained(self, node, img_shape):
        img_shape = img_shape[2:] if len(img_shape) == 5 else img_shape
        voxel = self.graph.node_voxel(node)
        return img_util.is_patch_contained(voxel, self.radial_shape, img_shape)

    def migrate_result(self, name):
        # Set paths
        src_bucket, src_prefix = util.parse_cloud_path(output_gcs_dir)
        src_path = os.path.join(src_prefix, name)

        dst_bucket, dst_prefix = util.parse_cloud_path(output_s3_dir)
        dst_path = os.path.join(dst_prefix, name)

        # Migrate
        asyncio.run(
            util.migrate_omezarr_gcs_to_s3(
                src_bucket,
                src_path,
                dst_bucket,
                dst_path,
            )
        )

    def node_to_slices(self, node):
        voxel = self.graph.node_voxel(node)
        slices = img_util.get_slices(voxel, self.radial_shape)
        return (0, 0, *slices)

    def traverse_graph(self):
        for nodes in map(list, nx.connected_components(self.graph)):
            yield from self.traverse_connected_component(nodes[0])

    def traverse_connected_component(self, root):
        queue = [(root, np.inf)]
        visited = set(queue)
        while queue:
            # Visit node
            i, dist_i = queue.pop()
            if dist_i >= self.step_size or self.graph.degree[i] == 1:
                yield i
                dist_i = 0

            # Update queue
            for j in self.graph.neighbors(i):
                if j not in visited:
                    dist_j = dist_i + self.graph.dist(i, j)
                    queue.append((j, dist_j))
                    visited.add(j)
        yield i


# --- Helpers ---
def load_skeletons():
    """
    Loads the SWC files into a SkeletonGraph.

    Parameters
    ----------
    swc_pointer : str
        Path to SWC files to be loaded.

    Returns
    -------
    graph : SkeletonGraph
        Graph with specified SWC files loaded.
    """
    gt_graph = SkeletonGraph()
    for swc_name in gt_swc_names:
        swc_path = os.path.join(gt_swc_dir, swc_name)
        gt_graph.load(swc_path)
    return gt_graph


def write_list_to_gcs(bucket_name, blob_name, data):
    # Initializations
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Convert list to newline-separated string
    text = "\n".join(map(str, data))
    blob.upload_from_string(text, content_type="text/plain")


if __name__ == "__main__":
    # Parameters
    brain_id = "802449"
    is_test = True
    radial_shape = (32, 32, 32) if is_test else (512, 512, 512)
    step_size = 10 if is_test else 128

    # Paths
    if is_test:
        num_levels = 3
        gt_swc_names = ["00005.swc", "00013.swc"]
        gt_swc_dir = "gs://allen-nd-goog/from_aind/training-data_2025-07-30/swcs/block_000/"
        input_img_path = "gs://allen-nd-goog/from_aind/training-data_2025-07-30/blocks/block_000/input.zarr/0"
        output_gcs_dir = "gs://allen-nd-goog/from_aind/agrim-experimental/image-carveouts/754612/blocks/block_000/"
        output_s3_dir = "s3://aind-msma-morphology-data/anna.grim/image-carveouts/754612/blocks/block_000/"
    else:
        num_levels = 7
        gt_swc_names = ["N002-802449-PP.swc"]
        gt_swc_dir = f"gs://allen-nd-goog/ground_truth_tracings/{brain_id}/voxel"
        input_img_path = os.path.join(img_util.find_img_path("allen-nd-goog", "from_aind/", brain_id), str(0))
        output_gcs_dir = f"gs://allen-nd-goog/from_aind/agrim-experimental/image-carveouts/{brain_id}/whole-brain"
        output_s3_dir = f"s3://aind-msma-morphology-data/anna.grim/image-carveouts/{brain_id}/whole-brain"
        assert brain_id in input_img_path

    # Run code
    main()
