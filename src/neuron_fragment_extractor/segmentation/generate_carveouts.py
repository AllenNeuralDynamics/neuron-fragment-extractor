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
from neuron_fragment_extractor.utils.img_util import TensorStoreReader
from neuron_fragment_extractor.utils import img_util


def main():
    # Paths
    carveout_raw_path = os.path.join(output_dir, "input.zarr")
    carveout_mask_path = os.path.join(output_dir, "mask.zarr")

    # Initializations
    gt_graph = load_skeletons()
    src_img = TensorStoreReader(img_path)
    dst_img = np.zeros(src_img.shape())
    dst_mask = np.zeros(src_img.shape())
    print("\ncarveout_img.shape:", dst_img.shape)

    # Generate carveouts
    carve_out_pipeline = CarveOutPipeline(gt_graph, radial_shape)
    carve_out_pipeline.generate_raw(src_img, dst_img)
    carve_out_pipeline.generate_mask(dst_mask)

    tifffile.imwrite("/home/jupyter/raw.tiff", dst_img[0, 0].astype(np.uint16))
    tifffile.imwrite("/home/jupyter/mask.tiff", dst_mask[0, 0].astype(np.uint8))

    # Write metadata


class CarveOutPipeline:

    def __init__(
        self,
        graph,
        radial_shape,
        num_readers=1,
        num_writers=1,
        prefetch=64,
        step_size=10
    ):
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
                if self.is_patch_contained(node, dst_img.shape[2:]):  # TEMP
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
                dst_img[slices] = patch
                pbar.update(1)

        # Initializations
        slices_q = queue.Queue(maxsize=self.prefetch)
        patch_q = queue.Queue(maxsize=self.prefetch)
        pbar = tqdm(total=self.count_patches(dst_img.shape[2:]), desc="Raw")
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
            for node in self.traverse_graph():
                if self.is_patch_contained(node, dst_img.shape[2:]):  # TEMP
                    slices_q.put(self.node_to_slices(node))
            for _ in range(self.num_readers):
                slices_q.put(stop)

        def writer():
            finished_readers = 0
            while True:
                item = slices_q.get()
                if item is stop:
                    finished_readers += 1
                    if finished_readers == self.num_readers:
                        break
                    continue

                dst_img[item] = np.ones(self.radial_shape, dtype=int)
                pbar.update(1)

        # Initializations
        pbar = tqdm(total=self.count_patches(dst_img.shape[2:]), desc="Mask")
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
def load_skeletons():
    gt_graph = SkeletonGraph()
    for swc_name in gt_swc_names:
        swc_path = os.path.join(gt_dir, swc_name)
        gt_graph.load(swc_path)
    return gt_graph


if __name__ == "__main__":
    # Parameters
    brain_id = "802449"
    gt_swc_names = ["00001.swc", "00002.swc"]  #["N002-802449-PP.swc"]
    radial_shape = (30, 30, 30)   #(512, 512, 512)
    step_size = 10

    # Paths
    #gt_dir = f"gs://allen-nd-goog/ground_truth_tracings/{brain_id}/voxel"
    #img_path = "gs://allen-nd-goog/from_aind/exaSPIM_802449_2025-12-16_18-17-47_training-data/whole-brain/fused.zarr/0"
    output_dir = f"gs://allen-nd-goog/from_aind/agrim-experimental/image_carveouts/{brain_id}"

    gt_dir = "gs://allen-nd-goog/from_aind/training-data_2025-07-30/swcs/block_000/"
    img_path = "gs://allen-nd-goog/from_aind/training-data_2025-07-30/blocks/block_000/input.zarr/0"

    # Run code
    main()
