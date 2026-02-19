"""
Created on Mon Feb 2 12:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that extracts nonmerge sites from fragments that intersect with ground
truth neuron tracings. The following outputs are extracted:
    /{output_dir}
        * nonmerge_sites.csv
        * nonmerge_sites.zip
"""

from scipy.spatial import KDTree
from tqdm import tqdm

import argparse
import numpy as np
import os
import pandas as pd

from neuron_fragment_extractor.graph_classes import SkeletonGraph
from neuron_fragment_extractor.utils import swc_util


def main():
    # Extract non-merge sites
    graph = load_fragments()
    sites = find_nonmerge_sites(graph)
    site_locations = [graph.midpoint(*tuple(site)) for site in sites]

    # Save results
    zip_path = os.path.join(output_dir, "nonmerge_sites.zip")
    swc_util.write_points(site_locations, zip_path, prefix="nonmerge-")
    save_site_metadata(graph, sites)
    print("# Non-Merge Sites:", len(sites))


def load_fragments():
    # Paths
    fragments_path = os.path.join(output_dir, "fragments_nonmerged.zip")
    gt_path = os.path.join(output_dir, "gt_neurons.zip")

    # Load skeletons
    graph = load_skeletons(fragments_path)
    gt_graph = load_skeletons(gt_path)
    graph.clip_to_groundtruth(gt_graph, 50)
    return graph


def find_nonmerge_sites(graph):
    sites = find_nearby_sites(graph)
    sites = filter_with_nms(graph, sites)
    sites = filter_nearby_leafs(graph, sites)
    return sites


def save_site_metadata(graph, sites):
    metadata = list()
    for cnt, (i, j) in enumerate(sites):
        metadata.append(
            {
                "NonMerge_ID": f"nonmerge-{cnt + 1}.swc",
                "Segment_ID_1": graph.get_node_segment_id(i),
                "Segment_ID_2": graph.get_node_segment_id(j),
                "Voxel_1": tuple([int(u) for u in graph.node_voxel(i)]),
                "Voxel_2": tuple([int(u) for u in graph.node_voxel(j)]),
                "World_1": tuple([float(u) for u in graph.node_xyz[i]]),
                "World_2": tuple([float(u) for u in graph.node_xyz[j]]),
            }
        )
    path = os.path.join(output_dir, "nonmerge_sites.csv")
    pd.DataFrame(metadata).to_csv(path, index=False)


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
    for root in tqdm(sites, "Filter"):
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
    for i in tqdm(graph.nodes, desc="Search"):
        for j in graph.get_nearby_nodes(graph.node_xyz[i], 20):
            if graph.node_component_id[i] != graph.node_component_id[j]:
                nodes.add(frozenset({i, j}))
                break
    return list(nodes)


def get_site_component_ids(graph, site):
    i, j = site
    id_i = graph.node_component_id[i]
    id_j = graph.node_component_id[j]
    return frozenset({id_i, id_j})


def load_skeletons(swc_pointer):
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
    graph = SkeletonGraph(anisotropy=(0.748, 0.748, 1.0))
    graph.load(swc_pointer)
    return graph


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir")

    # Run code
    args = parser.parse_args()
    output_dir = args.output_dir

    main()
