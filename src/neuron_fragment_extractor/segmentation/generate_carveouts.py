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
    write_ome_ngff_metadata,
)
from google.cloud import storage
from threading import Lock, Thread
from tqdm import tqdm

import asyncio
import networkx as nx
import numpy as np
import os
import queue

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
        gt_graph,
        img_shape,
        radial_shape,
        num_levels=num_levels,
        step_size=step_size,
    )
    pipeline("mask.zarr")
    pipeline("input.zarr", src_img=src_img)

    # Write metadata
    metadata = ["SWC Names"] + input_swc_names
    bucket, prefix = util.parse_cloud_path(output_gcs_dir)
    blob_name = os.path.join(prefix, "swc_names.txt")
    write_list_to_gcs(bucket, blob_name, metadata)


class CarveOutPipeline:

    def __init__(
        self,
        graph,
        img_shape,
        radial_shape,
        chunks=(1, 1, 128, 128, 128),
        num_levels=1,
        num_workers=32,
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
        num_workers : int, optional
            Number of workers used to read and write image patches.
        prefetch : int, optional
            Number of image patches to be prefeteched. Default is 128.
        step_size : float, optional
            Distance (in microns) between carved-out regions, measured along
            graph traversal. Default is 20.
        """
        # Check inputs
        assert len(img_shape) == 5, "Image shape must have format (T,C,Z,Y,X)"

        # Instance attributes
        self.chunks = chunks
        self.img_shape = img_shape
        self.num_levels = num_levels
        self.num_workers = num_workers
        self.prefetch = prefetch
        self.radial_shape = radial_shape
        self.voxel_size = voxel_size

        # Core data structures
        self.graph = graph
        self.centers = self.list_centers(step_size)[0:512]

    def __call__(self, filename, src_img=None):
        # Create and store the array
        print(f"\nStep 1: Create OME-Zarr with shape={self.img_shape}")
        root_path = os.path.join(output_gcs_dir, filename)
        spec = self.get_tensorstore_spec(root_path, level=0)
        dst_img = TensorStoreImage(spec=spec)

        # Generate carve-out
        print("Step 2: Generate Image Carve-Out")
        if filename == "mask.zarr":
            self.generate_mask(dst_img)
        else:
            self.generate_raw(src_img, dst_img)

        # Generate image pyramid
        print("Step 3: Generate Image Pyramid")
        self.generate_pyramid(root_path)

        # Write metadata
        print("Step 4: Write MetaData")
        #write_ome_ngff_metadata()

        # Migrate result
        print("Step 5: Migrating from GCS to S3")
        # self.migrate_result(filename)

    def generate_mask(self, dst_img):
        """
        Generates a binary mask that indicates which voxels are contained in
        the image carve-out.
        """
        def worker():
            """
            Writes an array of ones to the mask (i.e. dst_img).
            """
            while True:
                # Get slice
                slices = slices_queue.get()
                if slices is None:
                    break

                # Write patch
                with write_lock:
                    dst_img.write(mask_patch, slices)
                pbar.update(1)

        # Initializations
        mask_patch = np.ones(self.radial_shape, dtype=np.uint16)
        pbar = tqdm(total=len(self.centers), desc="   Mask")
        write_lock = Lock()

        # Start workers
        slices_queue = queue.Queue(maxsize=self.prefetch)
        threads = [Thread(target=worker) for _ in range(self.num_workers)]
        for t in threads:
            t.start()

        # Populate queue
        for node in self.centers:
            slices_queue.put(self.node_to_slices(node))
        for _ in range(self.num_workers):
            slices_queue.put(None)

        # Wait until completed
        for t in threads:
            t.join()

    def generate_raw(self, src_img, dst_img):
        """
        Generates an image carve out by copying image patches from "src_img"
        to "dst_img" that are centered about the skeleton.
        """
        def worker():
            while True:
                slices = slices_queue.get()
                if slices is None:
                    break

                patch = src_img.read(slices)
                with write_lock:
                    dst_img.write(patch, slices)
                pbar.update(1)

        # Initializations
        pbar = tqdm(total=len(self.centers), desc="   Raw")
        write_lock = Lock()

        # Start workers
        slices_queue = queue.Queue(maxsize=self.prefetch)
        threads = [Thread(target=worker) for _ in range(self.num_workers)]
        for t in threads:
            t.start()

        # Populate queue
        for node in self.centers:
            slices_queue.put(self.node_to_slices(node))
        for _ in range(self.num_workers):
            slices_queue.put(None)

        # Wait until completion
        for t in threads:
            t.join()

    def generate_pyramid(self, root_path):
        """
        Generate OME-Zarr pyramid using TensorStore.

        Parameters
        ----------
        img_path : str
            Path to highest resolution image that downsampled versions are
            generated from.
        """
        for level in range(1, num_levels):
            # Set source image
            src_path = os.path.join(root_path, str(level - 1))
            src = TensorStoreImage(img_path=src_path)

            # Set dst image
            dst_spec = self.get_tensorstore_spec(root_path, level=level)
            dst = TensorStoreImage(spec=dst_spec)
            dst_shape = [s // 2**level for s in self.radial_shape]

            # Generate downsampled
            self._create_pyramid_level(src, dst, dst_shape, level)

    def _create_pyramid_level(self, src, dst, dst_shape, level):
        def worker():
            while True:
                # Get slices
                node = slices_queue.get()
                if node is None:
                    break

                # Read
                read_slices = self.node_to_slices(node, level=level-1)
                patch = src.read(read_slices)
                patch = img_util.resize(patch, dst_shape)

                # Write
                write_slices = self.node_to_slices(node, level=level)
                with write_lock:
                    dst.write(patch, write_slices)
                pbar.update(1)

        # Initializations
        pbar = tqdm(total=len(self.centers), desc=f"   Level {level}")
        write_lock = Lock()

        # Start threads
        slices_queue = queue.Queue(maxsize=self.prefetch)
        threads = [Thread(target=worker) for _ in range(self.num_workers)]
        for t in threads:
            t.start()

        # Populate queue
        for node in self.centers:
            slices_queue.put(node)
        for _ in range(self.num_workers):
            slices_queue.put(None)

        # Wait until completion
        for t in threads:
            t.join()

    # --- Helpers ---
    def get_tensorstore_spec(self, root_path, level=0):
        # Extract info
        bucket_name, prefix = util.parse_cloud_path(root_path)
        shape = (1, 1, *(s // 2**level for s in self.img_shape[2:]))
        chunks = (1, 1, *(s // 2**level for s in self.chunks[2:]))

        # Create spec
        spec = {
            "driver": "zarr2",
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

    def is_patch_contained(self, node):
        voxel = self.graph.node_voxel(node)
        is_contained = img_util.is_patch_contained(
            voxel, self.radial_shape, self.img_shape[2:]
        )
        return is_contained

    def list_centers(self, step_size):
        """
        Generates nodes along skeletons used to create the image carve out.

        Parameters
        ----------
        step_size : float
            Distance (in microns) between carved-out regions, measured along
            graph traversal.

        Returns
        -------
        Iterator[int]
            Node IDs used to create image carve out.
        """
        centers = list()
        for nodes in map(list, nx.connected_components(self.graph)):
            root = nodes[0]
            queue = [(root, np.inf)]
            visited = set(queue)
            while queue:
                # Visit node
                i, dist_i = queue.pop()
                if dist_i >= step_size or self.graph.degree[i] == 1:
                    centers.append(i)
                    dist_i = 0

                # Update queue
                for j in self.graph.neighbors(i):
                    if j not in visited:
                        dist_j = dist_i + self.graph.dist(i, j)
                        queue.append((j, dist_j))
                        visited.add(j)
            centers.append(i)
        return [c for c in centers if self.is_patch_contained(c)]

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

    def node_to_slices(self, node, level=0):
        voxel = [u // 2**level for u in self.graph.node_voxel(node)]
        shape = [s // 2**level for s in self.radial_shape]
        slices = img_util.get_center_slices(voxel, shape)
        return slices


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
    for swc_name in input_swc_names:
        swc_path = os.path.join(input_swc_dir, swc_name)
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
    is_single_tracing = True
    is_test = True

    input_swc_names = (
        ["00005.swc"] if is_test else ["N002-802449-PP.swc"]
    )
    num_levels = 3 if is_test else 7
    radial_shape = (32, 32, 32) if is_test else (512, 512, 512)
    step_size = 16 if is_test else 256

    # Check whether to add neuron ID to output dir
    if is_single_tracing and not is_test:
        assert len(input_swc_names) == 1
        neuron_id = input_swc_names[0][0:4]
    else:
        neuron_id = ""

    # Paths
    if is_test:
        input_swc_dir = "gs://allen-nd-goog/from_aind/training-data_2025-07-30/swcs/block_000/"
        input_img_path = "gs://allen-nd-goog/from_aind/training-data_2025-07-30/blocks/block_000/input.zarr/0"
        output_gcs_dir = "gs://allen-nd-goog/from_aind/agrim-experimental/image-carveouts/754612/blocks/block_000/"
        output_s3_dir = "s3://aind-msma-morphology-data/anna.grim/image-carveouts/754612/blocks/block_000/"
    else:
        input_swc_dir = f"gs://allen-nd-goog/ground_truth_tracings/{brain_id}/voxel/"
        input_img_path = os.path.join(img_util.find_img_path("allen-nd-goog", "from_aind/", brain_id), str(0))
        output_gcs_dir = f"gs://allen-nd-goog/from_aind/agrim-experimental/image-carveouts/{brain_id}/whole-brain-{neuron_id}/"
        output_s3_dir = f"s3://aind-msma-morphology-data/anna.grim/image-carveouts/{brain_id}/whole-brain-{neuron_id}/"
        assert brain_id in input_img_path

    # Run code
    main()
