"""
Created on Tue Mar 3 14:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for extracting regions of ground truth skeletons that intersect with
image carve-outs

"""

from segmentation_skeleton_metrics.data_handling.graph_loading import (
    DataLoader,
    LabelHandler
)
from segmentation_skeleton_metrics.utils.img_util import TensorStoreReader

import networkx as nx


def main():
    for graph in load_skeletons():
        remove_excluded_nodes(graph)
        remove_small_components(graph)
        graph.write_swcs(output_dir)


def load_skeletons():
    label_handler = LabelHandler()
    mask = TensorStoreReader(mask_path)
    dataloader = DataLoader(label_handler)
    graphs = dataloader.load_groundtruth(gt_swcs_path, mask)
    return graphs.values()


def remove_excluded_nodes(graph):
    nodes = graph.get_nodes_with_label(0)
    graph.remove_nodes_from(nodes)


def remove_small_components(graph):
    nodes_to_remove = set()
    for nodes in nx.connected_components(graph):
        if len(nodes) < 20:
            nodes_to_remove = nodes_to_remove.union(nodes)
    graph.remove_nodes_from(nodes_to_remove)


if __name__ == "__main__":
    # Parameters
    brain_id = "802449"
    is_test = True

    # Paths
    if is_test:
        gt_swcs_path = "gs://allen-nd-goog/from_aind/training-data_2025-07-30/swcs/block_000/"
        mask_path = "s3://aind-msma-morphology-data/anna.grim/image-carveouts/754612/blocks/block_000/mask.zarr/0"
        output_dir = "/home/jupyter/results/carveout_skeletons/test"
    else:
        gt_swcs_path = f"gs://allen-nd-goog/ground_truth_tracings/{brain_id}/voxel"
        mask_path = f"s3://aind-msma-morphology-data/anna.grim/image-carveouts/{brain_id}/whole-brain/mask.zarr/0"
        output_dir = "/home/jupyter/results/carveout_skeletons/test"

    # Run code
    main()
