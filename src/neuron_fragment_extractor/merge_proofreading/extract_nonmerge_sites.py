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

from scipy.spatial import KDTree
from tqdm import tqdm

import numpy as np
import os

from neuron_fragment_extractor.graph_classes import SkeletonGraph
from neuron_fragment_extractor.utils import swc_util


def main():
    # Paths
    fragments_path = os.path.join(output_dir, "fragments_nonmerged.zip")
    gt_path = os.path.join(output_dir, "gt_neurons.zip")

    # Load skeletons
    graph = load_skeletons(fragments_path)
    gt_graph = load_skeletons(gt_path)
    graph.clip_to_groundtruth(gt_graph, 50)

    # Detect
    sites = list()
    sites.extend(find_nearby_branches(graph))
    #sites.update(find_branching_points(graph, gt_graph))
    site_locations = [graph.midpoint(*tuple(site)) for site in sites]

    # Save results
    zip_path = os.path.join(output_dir, "nonmerge_sites.zip")
    swc_util.write_points(site_locations, zip_path, prefix="nonmerge-")


def find_nearby_branches(graph):
    sites = find_nearby_sites(graph)
    sites = filter_with_nms(graph, sites)
    sites = filter_nearby_leafs(graph, sites)
    return sites


def find_branching_points(graph, gt_graph):
    # Build KD-Tree with GT branching points

    # Parse branching points in fragments graph
    pass


# --- Helpers ---
def filter_with_nms(graph, sites):
    # Sort sites by distance
    dists = [graph.dist(i, j) for i, j in sites]
    sites = [sites[i] for i in np.argsort(dists)]

    # Build KD-Tree with approximate site locations
    midpoints = [graph.midpoint(i, j) for i, j in sites]
    kdtree = KDTree(midpoints)

    # NMS
    visited = set()
    filtered_sites = list()
    for root in sites:
        # Set local min
        if root not in visited:
            filtered_sites.append(root)
            visited.add(root)
        else:
            continue

        # Suppress neighborhood
        root_component_ids = get_site_component_ids(graph, root)
        root_xyz = graph.midpoint(*tuple(root))
        for idx in kdtree.query_ball_point(root_xyz, 70):
            component_ids = get_site_component_ids(graph, sites[idx])
            is_same_components = root_component_ids == component_ids
            if is_same_components and sites[idx] not in visited:
                visited.add(sites[idx])
    return filtered_sites


def filter_nearby_leafs(graph, sites):
    filtered_sites = list()
    for site in sites:
        i, j = site
        if graph.near_leaf(i, 30) and graph.near_leaf(j, 30):
            continue
        else:
            filtered_sites.append(site)
    return filtered_sites


def find_nearby_sites(graph):
    nodes = set()
    for i in tqdm(graph.nodes):
        for j in graph.get_nearby_nodes(graph.node_xyz[i], radius):
            if graph.node_component_id[i] != graph.node_component_id[j]:
                nodes.add(frozenset({i, j}))
                break
    return list(nodes)


def get_site_component_ids(graph, site):
    i, j = site
    id_i = graph.node_component_id[i]
    id_j = graph.node_component_id[j]
    return frozenset({id_i, id_j})


def load_skeletons(swcs_pointer):
    graph = SkeletonGraph()
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
