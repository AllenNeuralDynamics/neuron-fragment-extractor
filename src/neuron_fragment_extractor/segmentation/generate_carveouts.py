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

from google.cloud import storage
from threading import Lock, Thread
from tqdm import tqdm

import asyncio
import gcsfs
import json
import networkx as nx
import numpy as np
import os
import queue

from neuron_fragment_extractor.graph_classes import SkeletonGraph
from neuron_fragment_extractor.utils.img_util import TensorStoreImage
from neuron_fragment_extractor.utils import img_util, util


def main():
    """
    Execute the full image carve-out generation pipeline.
    """
    # Initializations
    gt_graph = load_skeletons()
    src_img = TensorStoreImage(input_img_path)

    # Generate carveouts
    pipeline = CarveOutPipeline(
        gt_graph,
        src_img.shape(),
        radial_shape,
        chunks=chunks,
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
    """
    Pipeline for generating skeleton-centered image carve-outs and
    multiscale OME-Zarr pyramids from large volumetric images.
    """

    def __init__(
        self,
        graph,
        img_shape,
        radial_shape,
        chunks=(1, 1, 128, 256, 256),
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
        img_shape : Tuple[int]
            Shape of image carve out.
        radial_shape : Tuple[int]
            Shape of region centered about skeleton to be carved out.
        chunks : Tuple[int], optional
            Chunk shape used to write OME-Zarr image. Default is (1, 1, 128
            128, 128).
        num_workers : int, optional
            Number of workers used to read and write image patches. Default is
            32.
        prefetch : int, optional
            Number of image patches to be prefeteched. Default is 128.
        step_size : float, optional
            Distance (in microns) between carved-out regions, measured along
            graph traversal. Default is 20.
        voxel_size : Tuple[float], optional
            Physical voxel size for the highest resolution level. Default is
            (1.0, 0.748, 0.748).
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
        self.centers = self.list_centers(step_size)[0:64]

    def __call__(self, filename, src_img=None):
        """
        Executes the full carve-out workflow for a given output file, which
        involves the following steps:
            1. Create an OME-Zarr array and write ".zattrs".
            2. Generate image carve-out
            3. Generate a multiscale image pyramid.
            4. Migrate the resulting Zarr directory to another storage
               location such as S3.

        Parameters
        ----------
        filename : str
            Name of the output Zarr directory.
        src_img : TensorStoreImage, optional
            Source image to read from when generating raw image carve-outs.
        """
        # Create and store the array
        print(f"\nStep 1: Create OME-Zarr with shape={self.img_shape}")
        root_path = os.path.join(output_gcs_dir, filename)
        spec = self.get_tensorstore_spec(root_path, level=0)
        dst_img = TensorStoreImage(spec=spec)
        self.write_zattrs(root_path)

        # Generate carve-out
        print("Step 2: Generate Image Carve-Out")
        if filename == "mask.zarr":
            self.generate_mask(dst_img)
        else:
            self.generate_raw(src_img, dst_img)

        # Generate image pyramid
        print("Step 3: Generate Image Pyramid")
        self.generate_pyramid(root_path)

        # Migrate result
        print("Step 4: Migrating from GCS to S3")
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

    def generate_raw(self, src, dst):
        """
        Generates an image carve-out by copying patches centered on the
        skeleton from the source image to the destination image.

        Parameters
        ----------
        src : TensorStoreImage
            Image to be read from.
        dst : TensorStoreImage
            Image to be written to.
        """
        def worker():
            """
            Reads an array from the source image, then writes it to the
            destination image.
            """
            while True:
                slices = slices_queue.get()
                if slices is None:
                    break

                patch = src.read(slices)
                with write_lock:
                    dst.write(patch, slices)
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
        root_path : str
            Path to root directory of the OME-Zarr image.
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
        """
        Generate a single level of the image pyramid by downsampling patches
        from the previous level and writing them to the destination level.

        Parameters
        ----------
        src : TensorStoreImage
            Source image corresponding to pyramid level "level - 1" from
            which patches are read.
        dst : TensorStoreImage
            Destination image corresponding to pyramid level "level" to
            which resized patches are written.
        dst_shape : Tuple[int]
            Target shape of each patch written at this pyramid level.
        level : int
            Pyramid level being generated.
        """
        def worker():
            """
            Writes downsampled patch to the destination image.
            """
            while True:
                # Get slices
                node = slices_queue.get()
                if node is None:
                    break

                # Read
                read_slices = self.node_to_slices(node, level=level-1)
                patch = src.read(read_slices)
                patch = img_util.resize(patch, dst_shape).astype(np.uint16)

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
        shape = (1, 1, *(s // 2**level for s in self.img_shape[2:]))
        chunks = (1, 1, *(max(s // 2**level, 32) for s in self.chunks[2:]))

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

    def is_patch_contained(self, node):
        """
        Checks whether the patch centered at a given node lies fully within
        the spatial bounds of the image.

        Parameters
        ----------
        node : int
            Node ID whose associated voxel location is used as the center of
            the patch.

        Returns
        -------
        bool
            True if the patch centered at the node is entirely contained
            within the image bounds, otherwise False.
        """
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
        """
        Migrates the image carve out from GCS to S3.

        Parameters
        ----------
        name : str
            Name of the OME-Zarr image.
        """
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
        """
        Converts a node to a set of array slices for extracting a patch
        from a specific pyramid level of the image.

        Parameters
        ----------
        node : int
            Node ID whose voxel coordinate are used as the center of the
            patch.
        level : int, optional
            Pyramid level for which slices are computed. Default is 0.

        Returns
        -------
        List[slice]
            Slice objects specifying the spatial region of the patch centered
            at the node for the requested pyramid level.
        """
        voxel = [u // 2**level for u in self.graph.node_voxel(node)]
        shape = [s // 2**level for s in self.radial_shape]
        slices = img_util.get_center_slices(voxel, shape)
        return slices

    def write_zattrs(self, root_path):
        """
        Writes the ".zattrs" metadata file for an OME-Zarr dataset to GCS.

        Parameters
        ----------
        root_path : str
            GCS path to the root of the Zarr directory where ".zattrs" should
            be written.
        """
        fs = gcsfs.GCSFileSystem(project="allen-nd-goog")
        with fs.open(f"{root_path}/.zattrs", "w") as f:
            zattrs = img_util.create_zattrs(self.num_levels)
            f.write(json.dumps(zattrs, indent=4))


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
    """
    Writes a list to a Google Cloud Storage (GCS) object as a text file.

    Parameters
    ----------
    bucket_name : str
        Name of the GCS bucket where the blob will be stored.
    blob_name : str
        Path or name of the blob within the bucket.
    data : list
        List of items to write.
    """
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
    chunks = (1, 1, 256, 256, 256) if is_test else (1, 1, 512, 512, 512)
    num_levels = 3 if is_test else 7
    radial_shape = (32, 32, 32) if is_test else (512, 512, 512)
    step_size = 4 if is_test else 64

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
