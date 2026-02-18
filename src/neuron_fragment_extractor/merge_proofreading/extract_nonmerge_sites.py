"""
Created on Mon Feb 2 12:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that extracts nonmerge sites from fragments that intersect with ground
truth neuron tracings. The following outputs are extracted:
    /{brain_id}
        /{segmentation_id}
            * nonmerge_sites.csv
            * nonmerge_sites.zip
"""

from neuron_fragment_extractor.graph_classes import SkeletonGraph
from tqdm import tqdm

import numpy as np
import os


def main():
    # Paths
    fragments_path = os.path.join(output_dir, "fragments_nonmerged.zip")
    gt_path = os.path.join(output_dir, "gt_neurons.zip")

    # Load skeletons
    fragments_graph = load_skeletons(fragments_path)
    gt_graph = load_skeletons(gt_path)

    # Detect
    nonmerge_sites = dict()
    nonmerge_sites.update(find_nearby_branches(fragments_graph))
    nonmerge_sites.update(find_branching_points(fragments_graph, gt_graph))


def find_nearby_branches(graph):
    # Find sorted sites
    nodes = find_nearby_sites(graph)
    dists = [graph.dist(i, j) for i, j in nodes]
    nodes = [nodes[i] for i in np.argsort(dists)]
    print("# Initial Sites:", len(nodes))

    # Prune redundant sites
    visited = set()


def find_branching_points(fragments_graph, gt_graph):
    pass


# --- Helpers ---
def find_nearby_sites(graph):
    nodes = set()
    for i in tqdm(graph.nodes):
        for j in graph.get_nearby_nodes(graph.node_xyz[i], radius):
            if graph.node_component_id[i] != graph.node_component_id[j]:
                nodes.add(frozenset({i, j}))
                break
    return list(nodes)


def load_skeletons(swcs_pointer):
    graph = SkeletonGraph(verbose=True)
    graph.load(swcs_pointer)
    return graph


if __name__ == "__main__":
    # Parameters
    brain_id = "802449"
    segmentation_id = "jin_masked_mean40_stddev105"

    parameters = (128, 128, 128)
    radius = 20

    # Paths
    output_dir = f"/home/jupyter/results/merge_datasets/{brain_id}/{segmentation_id}"

    # Run code
    main()
