"""
Created on Wed August 28 17:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Implementation of custom subclasses of NetowrkX called "SkeletonGraph",
"GroundTruthGraph", and "TracedGraph" which are used to represent neuron
morphologies.

"""

from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean

import networkx as nx
import numpy as np

from neuron_fragment_extractor.utils import graph_util


class SkeletonGraph(nx.Graph):

    def __init__(self, anisotropy=(1.0, 1.0, 1.0), verbose=True):
        # Call parent class
        super().__init__()

        # Instance attributes
        self.anisotropy = anisotropy
        self.component_id_to_swc_name = dict()
        self.graph_loader = graph_util.GraphLoader()
        self.node_kdtree = None
        self.node_xyz = None
        self.verbose = verbose

    # --- Build Graph ---
    def load(self, swc_pointer):
        # Initialize node attribute data structures
        irreducibles = self.graph_loader(swc_pointer)
        num_nodes = graph_util.count_nodes(irreducibles)
        self.node_component_id = np.zeros((num_nodes), dtype=int)
        self.node_xyz = np.zeros((num_nodes, 3), dtype=np.float32)

        # Add irreducibles to graph
        component_id = 0
        while irreducibles:
            self.add_connected_component(irreducibles.pop(), component_id)
            component_id += 1

        # Build kd-tree from node coordinates
        self.node_kdtree = KDTree(self.node_xyz)

    def add_connected_component(self, irreducibles, component_id):
        """
        Adds a new connected component to the graph.

        Parameters
        ----------
        irreducibles : dict
            Dictionary with the following required fields:
                - "swc_id": SWC ID of the component.
                - "nodes": dictionary of node attributes.
                - "edges": dictionary of edge attributes.
        component_id : int
            Unique identifier for the connected component being added.
        """
        # Set component id
        self.component_id_to_swc_name[component_id] = irreducibles["swc_name"]

        # Add nodes
        node_id_mapping = self._add_nodes(irreducibles["nodes"], component_id)

        # Add edges
        for (i, j), attrs in irreducibles["edges"].items():
            edge_id = (node_id_mapping[i], node_id_mapping[j])
            self._add_edge(edge_id, attrs, component_id)

    def _add_nodes(self, node_dict, component_id):
        """
        Adds nodes to the graph from a dictionary of node attributes and
        returns a mapping from original node IDs to the new graph node IDs.

        Parameters
        ----------
        node_dict : dict
            Dictionary mapping original node IDs to their attributes. Each
            value must be a dictionary containing the key "xyz".
        component_id : str
            Connected component ID used to map node IDs back to SWC IDs via
            "self.component_id_to_swc_id".

        Returns
        -------
        node_id_mapping : Dict[int, int]
            Dictionary mapping the original node IDs from "node_dict" to the
            new node IDs assigned in the graph.
        """
        node_id_mapping = dict()
        for node_id, attrs in node_dict.items():
            new_id = self.number_of_nodes()
            self.node_xyz[new_id] = attrs["xyz"]
            self.node_component_id[new_id] = component_id
            self.add_node(new_id)
            node_id_mapping[node_id] = new_id
        return node_id_mapping

    def _add_edge(self, edge_id, attrs, component_id):
        """
        Adds an edge to the graph.

        Parameters
        ----------
        edge : Tuple[int]
            Edge to be added.
        attrs : dict
            Dictionary of attributes of "edge" obtained from an SWC file.
        component_id : int
            Connected component ID used to map node IDs back to SWC IDs via
            "self.component_id_to_swc_id".
        """
        # Determine orientation of attributes
        i, j = tuple(edge_id)
        dist_i = euclidean(self.node_xyz[i], attrs["xyz"][0])
        dist_j = euclidean(self.node_xyz[j], attrs["xyz"][0])
        if dist_i < dist_j:
            start = i
            end = j
        else:
            start = j
            end = i

        # Populate graph
        for cnt, xyz in enumerate(attrs["xyz"]):
            if cnt > 0 and cnt < len(attrs["xyz"]) - 1:
                # Add edge
                new_id = self.number_of_nodes()
                if cnt == 1:
                    self.add_edge(start, new_id)
                else:
                    self.add_edge(new_id, new_id - 1)

                # Store attributes
                self.node_xyz[new_id] = xyz
                self.node_component_id[new_id] = component_id
        self.add_edge(new_id, end)

    # --- Helpers ---
    def dist(self, i, j):
        """
        Computes the Euclidean distance between nodes "i" and "j".

        Parameters
        ----------
        i : int
            Node ID.
        j : int
            Node ID.

        Returns
        -------
        float
            Euclidean distance between nodes "i" and "j".
        """
        return euclidean(self.node_xyz[i], self.node_xyz[j])

    def get_nearby_nodes(self, xyz, radius):
        return self.node_kdtree.query_ball_point(xyz, radius)
