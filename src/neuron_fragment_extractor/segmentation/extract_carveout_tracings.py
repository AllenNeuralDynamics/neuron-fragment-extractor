"""
Created on Tue Mar 3 14:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for extracting regions of ground truth skeletons that intersect with
image carve-outs

"""

from segmentation_skeleton_metrics.data_handling.graph_loading import (
    GraphLoader,
    LabelHandler,
)
from segmentation_skeleton_metrics.utils.img_util import TensorStoreImage
from segmentation_skeleton_metrics.utils import util
from tqdm import tqdm

import networkx as nx


def main():
    util.mkdir(output_dir)
    for graph in tqdm(load_skeletons(), desc="Extract Skeletons"):
        remove_excluded_nodes(graph)
        remove_small_components(graph)
        graph.to_swcs(output_dir, use_color=False)


def load_skeletons():
    label_handler = LabelHandler()
    mask = TensorStoreImage(mask_path)
    graph_loader = GraphLoader(
        anisotropy=(0.748, 0.748, 1.0),
        fix_label_misalignments=False,
        is_groundtruth=True,
        label_handler=label_handler,
        label_mask=mask,
        use_anisotropy=True,
    )
    graphs = graph_loader(gt_swcs_path)
    return graphs.values()


def remove_excluded_nodes(graph):
    nodes = graph.nodes_with_label(0)
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

    # Paths
    gt_swcs_path = f"gs://allen-nd-goog/ground_truth_tracings/{brain_id}/world"
    mask_path = f"gs://allen-nd-goog/from_aind/agrim-experimental/image-carveouts/{brain_id}/whole-brain-N002/mask.zarr/0"
    output_dir = "/home/jupyter/results/carveout_skeletons/test"

    # Run code
    main()
