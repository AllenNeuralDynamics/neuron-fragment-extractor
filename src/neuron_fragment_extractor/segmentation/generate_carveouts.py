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

from tqdm import tqdm

import networkx as nx
import numpy as np
import os
import queue
import tifffile
import threading

from neuron_fragment_extractor.graph_classes import SkeletonGraph
from neuron_fragment_extractor.utils.img_util import TensorStoreImage
from neuron_fragment_extractor.utils import img_util


def main():
    # Initializations
    gt_graph = load_skeletons()
    src_img = TensorStoreImage(img_path)
    dst_img = init_carveout("input.zarr", src_img.shape())
    dst_mask = init_carveout("mask.zarr", src_img.shape())

    # Generate carveouts
    carve_out_pipeline = CarveOutPipeline(gt_graph, radial_shape)
    carve_out_pipeline.generate_raw(src_img, dst_img)
    carve_out_pipeline.generate_mask(dst_mask)

    # Add larger carve-out at soma

    #tifffile.imwrite("/home/jupyter/raw.tiff", dst_img[0, 0, 0:512].astype(np.uint16))
    #tifffile.imwrite("/home/jupyter/mask.tiff", dst_mask[0, 0, 0:512].astype(np.uint8))

    # Write metadata


class CarveOutPipeline:

    def __init__(
        self,
        graph,
        radial_shape,
        num_readers=32,
        num_writers=32,
        prefetch=128,
        step_size=20
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
        self.graph = graph
        self.num_readers = num_readers
        self.num_writers = num_writers
        self.prefetch = prefetch
        self.radial_shape = radial_shape
        self.step_size = step_size

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
                    break

                patch = src_img.read(slices)
                patch_q.put((slices, patch))

        def writer():
            finished_readers = 0
            while True:
                item = patch_q.get()
                if item is stop:
                    finished_readers += 1
                    if finished_readers == self.num_readers:
                        break
                    continue

                slices, patch = item
                dst_img.write(patch, slices)
                pbar.update(1)

        # Initializations
        slices_q = queue.Queue(maxsize=self.prefetch)
        patch_q = queue.Queue(maxsize=self.prefetch)
        pbar = tqdm(total=self.count_patches(dst_img.shape()), desc="Raw")
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
            while True:
                slices = slices_q.get()
                if slices is stop:
                    finished_readers += 1
                    if finished_readers == self.num_readers:
                        break
                    continue

                dst_img.write(mask_patch, slices)
                pbar.update(1)

        # Initializations
        pbar = tqdm(total=self.count_patches(dst_img.shape()), desc="Mask")
        mask_patch = np.ones(self.radial_shape, dtype=np.uint16)
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

        return dst_img  # TEMP

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
def init_carveout(filename, img_shape):
    """
    Initializes empty image that carve-out is to be written to.

    Parameters
    ----------
    filename : str
        Name of OME-Zarr image to be created.
    shape : Tuple[int]
        Shape of image to be created.

    Returns
    -------
    TensorStoreImage
        Empty image that carve-out is to be written to.
    """
    img_path = f"{output_dir}/{brain_id}/blocks/block_000/{filename}"
    img_util.init_omezarr_image(img_path, img_shape)
    return TensorStoreImage(os.path.join(img_path, str(0)))


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
        swc_path = os.path.join(gt_dir, swc_name)
        gt_graph.load(swc_path)
    return gt_graph


if __name__ == "__main__":
    # Parameters
    brain_id = "802449"
    gt_swc_names = ["00005.swc", "00013.swc"]  #["N002-802449-PP.swc"]
    radial_shape = (32, 32, 32)   #(512, 512, 512)
    step_size = 10

    # Paths
    #gt_dir = f"gs://allen-nd-goog/ground_truth_tracings/{brain_id}/voxel"
    #img_path = "gs://allen-nd-goog/from_aind/exaSPIM_802449_2025-12-16_18-17-47_training-data/whole-brain/fused.zarr/0"
    output_dir = "gs://allen-nd-goog/from_aind/agrim-experimental/image-carveouts"

    gt_dir = "gs://allen-nd-goog/from_aind/training-data_2025-07-30/swcs/block_000/"
    img_path = "gs://allen-nd-goog/from_aind/training-data_2025-07-30/blocks/block_000/input.zarr/0"

    # Run code
    main()
