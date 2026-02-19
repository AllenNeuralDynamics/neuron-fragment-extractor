"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that loads and preprocesses neuron tracings stored as SWC files. Plus,
helper routines for working with graphs.

"""

from collections import deque
from concurrent.futures import as_completed, ProcessPoolExecutor
from scipy.spatial.distance import euclidean

import multiprocessing
import networkx as nx
import numpy as np
import os

from neuron_fragment_extractor.utils import swc_util

os.environ["OPENBLAS_NUM_THREADS"] = "1"


class GraphLoader:
    """
    Class that loads SWC files and extracts the irreducible components of the
    corresponding graph.
    """

    def __init__(self, anisotropy=(1.0, 1.0, 1.0), prune_depth=24):
        """
        Builds graphs by reading swc files stored on either the cloud or local
        machine, then extracting the irreducible components.

        Parameters
        ----------
        anisotropy : List[float], optional
            Image to physical coordinates scaling factors to account for the
            anisotropy of the microscope. Default is (1.0, 1.0, 1.0).
        prune_depth : float, optional
                Branches with length less than "prune_depth" microns are
                pruned. Default is 24um.
        """
        # Instance attributes
        self.prune_depth = prune_depth
        self.swc_reader = swc_util.Reader(anisotropy)

    # --- Irreducibles Extraction ---
    def __call__(self, swc_pointer):
        """
        Processes a list of swc dictionaries in parallel and extracts the
        components of the irreducible subgraph from each. Note: this routine
        also breaks fragments that intersect multiple somas if soma locations
        are provided.

        Parameters
        ----------
        swc_pointer : Any
            Object that points to SWC files to be read.

        Returns
        -------
        irreducibles : List[dict]
            Dictionaries that contain components of the irreducible subgraph
            extracted from each SWC dictionary.
        """
        # Read SWC files
        swc_dicts = self.swc_reader(swc_pointer)

        # Load graphs
        multiprocessing.set_start_method('spawn', force=True)
        with ProcessPoolExecutor() as executor:
            # Start processes
            pending = set()
            for _ in range(min(512, len(swc_dicts))):
                pending.add(executor.submit(self.extract, swc_dicts.pop()))

            # Yield processes
            irreducibles = deque()
            while pending or swc_dicts:
                for process in as_completed(pending):
                    # Store completed processes
                    irreducibles.append(process.result())
                    pending.remove(process)

                    # Continue submitting processes
                    if swc_dicts:
                        pending.add(
                            executor.submit(self.extract, swc_dicts.pop())
                        )
        return irreducibles

    def extract(self, swc_dict):
        """
        Extracts the components of the irreducible subgraph from a given SWC
        dictionary.

        Parameters
        ----------
        swc_dict : dict
            Contents of an SWC file.

        Returns
        -------
        irreducibles_list : dict
            Dictionaries that each contains the irreducible components of a
             graph.
        """
        # Convert SWC file to graph
        graph = self.to_graph(swc_dict)

        # Extract irreducibes
        leaf = find_leaf(graph)
        irreducibles = self.get_irreducibles(graph, leaf)
        irreducibles["swc_name"] = graph.graph['swc_name']
        return irreducibles

    def get_irreducibles(self, graph, source):
        """
        Identifies irreducible components of a connected graph.

        Parameters
        ----------
        graph : networkx.Graph
            Graph to be searched.
        source : Set[int]
            Leaf nodes in the given graph.

        Returns
        -------
        irreducibles : dict
            Dictionary containing the irreducible components of a connected
            graph.
        leafs : Set[int]
            Leaf nodes that were visited.
        """

        def dist(i, j):
            """
            Computes distance between the given nodes.

            Parameters
            ----------
            i : int
                Node ID.
            j : int
                Node ID.

            Returns
            -------
            float
                Distance between nodes.
            """
            return euclidean(graph.graph["xyz"][i], graph.graph["xyz"][j])

        # Initializations
        irreducible_nodes = set({source})
        irreducible_edges = dict()

        # Main
        root = None
        for (i, j) in nx.dfs_edges(graph, source=source):
            # Check for start of irreducible edge
            if root is None:
                root, edge_length = i, 0
                attrs = {
                    "radius": [graph.graph["radius"][i]],
                    "xyz": [graph.graph["xyz"][i]],
                }

            # Visit node
            edge_length += dist(i, j)
            attrs["radius"].append(graph.graph["radius"][j])
            attrs["xyz"].append(graph.graph["xyz"][j])

            # Check for end of irreducible edge
            if graph.degree[j] != 2:
                # Set attrs
                irreducible_nodes.add(j)
                attrs = to_numpy(attrs)

                # Finish
                irreducible_edges[(root, j)] = attrs
                root = None

        # Store results
        irreducibles = {
            "nodes": set_node_attrs(graph, irreducible_nodes),
            "edges": set_edge_attrs(graph, irreducible_edges),
        }
        return irreducibles

    # --- Helpers ---
    def to_graph(self, swc_dict):
        """
        Builds a graph from a dictionary containing the contents of an SWC
        file.

        Parameters
        ----------
        swc_dict : dict
            Dictionaries whose keys and values are the attribute name and
            values from an SWC file.

        Returns
        -------
        graph : networkx.Graph
            Graph generated from "swc_dict".
        """
        graph = swc_util.to_graph(swc_dict, set_attrs=True)
        prune_branches(graph, self.prune_depth)
        return graph


# --- Helpers ---
def set_node_attrs(graph, nodes):
    """
    Extracts attributes for each node in the graph.

    Parameters
    ----------
    graph : networkx.Graph
        Graph that contains "nodes".
    nodes : List[int]
        Nodes whose attributes are to be extracted from the graph.

    Returns
    -------
    attrs : dict
        Dictionary where the keys are node ids and values are dictionaries
        containing the "radius" and "xyz" attributes of the nodes.
    """
    attrs = dict()
    for i in nodes:
        attrs[i] = {
            "radius": graph.graph["radius"][i], "xyz": graph.graph["xyz"][i]
        }
    return attrs


def set_edge_attrs(graph, attrs):
    """
    Sets the edge attributes of a given graph by updating node coordinates and
    resamples points in irreducible path.

    Parameters
    ----------
    graph : networkx.Graph
        Graph that attributes dictionary was built from.
    attrs : dict
        Dictionary where the keys are irreducible edge IDs and values are the
        corresponding attribute dictionaries.

    Returns
    -------
    attrs : dict
        Updated edge attribute dictionary.
    """
    for e in attrs:
        i, j = tuple(e)
        attrs[e]["xyz"][0] = graph.graph["xyz"][i]
        attrs[e]["xyz"][-1] = graph.graph["xyz"][j]
    return attrs


# --- Miscellaneous ---
def count_nodes(irreducibles):
    """
    Counts the total number of nodes represented in a collection of
    irreducible components.

    Parameters
    ----------
    irreducibles : List[Dict[str, numpy.ndarray]]
        List of dictionaries containing irreducible components. Each
        dictionary corresponds to a single with the following:
            - "nodes": list of irreducible nodes
            - "edges": list of paths between irreducible nodes

    Returns
    -------
    num_nodes : int
        Number of nodes contained in the given list of irreducible graph
        components.
    """
    num_nodes = 0
    for irr in irreducibles:
        num_nodes += len(irr["nodes"])
        for attrs in irr["edges"].values():
            num_nodes += len(attrs["xyz"]) - 2
    return num_nodes


def dist(graph, i, j):
    """
    Computes Euclidean distance between nodes i and j.

    Parameters
    ----------
    graph : netowrkx.Graph
        Graph containing nodes i and j.
    i : int
        Node ID.
    j : int
        Node ID.

    Returns
    -------
    float
        Euclidean distance between nodes i and j.
    """
    return euclidean(graph.graph["xyz"][i], graph.graph["xyz"][j])


def find_leaf(graph):
    """
    Finds a leaf node in the given graph.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched.

    Returns
    -------
    i : int
        Leaf node.
    """
    for i in graph.nodes:
        if graph.degree[i] == 1:
            return i


def get_component(graph, root):
    """
    Gets the connected component corresponding to "root" from "graph".

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched.
    root : int
        Node that breadth-first search starts from.

    Returns
    -------
    visited : Set[int]
        Nodes in the connected component corresponding to "root".
    """
    queue = [root]
    visited = set()
    while len(queue):
        i = queue.pop()
        visited.add(i)
        for j in [j for j in graph.neighbors(i) if j not in visited]:
            queue.append(j)
    return visited


def get_leafs(graph):
    """
    Gets the leaf nodes of the given graph.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched

    Returns
    -------
    List[int]
        Leaf nodes of the given graph.
    """
    return [i for i in graph.nodes if graph.degree[i] == 1]


def path_length(graph, max_length=np.inf):
    """
    Computes the path length of the given graph.

    Parameters
    ----------
    graph : networkx.Graph
        Graph whose nodes have an attribute called "xyz" which represents
        a 3d coordinate.
    max_length : float
        Maximum physical distance to search along the graph. Limits traversal
        depth and can improve performance.

    Returns
    -------
    length : float
        Path length of graph.
    """
    length = 0
    for i, j in nx.dfs_edges(graph):
        length += dist(graph, i, j)
        if length > max_length:
            break
    return length


def prune_branches(graph, depth):
    """
    Prunes branches with length less than "depth" microns.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched.
    depth : float
        Length of branches that are pruned.
    """
    for leaf in get_leafs(graph):
        branch = [leaf]
        length = 0
        for (i, j) in nx.dfs_edges(graph, source=leaf):
            # Visit edge
            length += dist(graph, i, j)
            if length > depth:
                break

            # Check whether to continue search
            if graph.degree(j) == 2:
                branch.append(j)
            elif graph.degree(j) > 2:
                graph.remove_nodes_from(branch)
                break


def to_numpy(attrs):
    """
    Converts edge attributes from a list to NumPy array.

    Parameters
    ----------
    attrs : dict
        Edge attribute dictionary.

    Returns
    -------
    attrs : dict
        Updated edge attribute dictionary.
    """
    attrs["xyz"] = np.array(attrs["xyz"], dtype=np.float32)
    attrs["radius"] = np.array(attrs["radius"], dtype=np.float16)
    return attrs
