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
    """
    A custom subclass of NetworkX tailored for graphs constructed from SWC
    files, where each connected component represents a single SWC file.
    """

    def __init__(self):
        """
        Instantiates a SkeletonGraph object.
        """
        # Call parent class
        super().__init__()

        # Instance attributes
        self.component_id_to_swc_name = dict()
        self.graph_loader = graph_util.GraphLoader()
        self.node_kdtree = None
        self.node_xyz = None

    # --- Build Graph ---
    def load(self, swc_pointer):
        """
        Loads SWC files into graph.

        Parameters
        ----------
        swc_pointer : str
            Path that points to SWC files to be read.
        """
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
    def clip_to_groundtruth(self, gt_graph, dist):
        """
        Removes nodes that are more than "dist" microns from "gt_graph".
    
        Parameters
        ----------
        gt_graph : SkeletonGraph
            Ground truth graph used as clipping reference.
        dist : float
            Distance threshold (in microns) that determines what nodes to remove.
        """
        # Remove nodes too far from ground truth
        d_gt, _ = gt_graph.node_kdtree.query(self.node_xyz)
        nodes = np.where(d_gt > dist)[0]
        self.remove_nodes_from(nodes)

        # Remove resulting small connected components
        for nodes in list(nx.connected_components(self)):
            if len(nodes) < 20:
                self.remove_nodes_from(nodes)
        self.relabel_nodes()

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
        """
        Gets node IDs within "radius" microns from the given point.

        Parameters
        ----------
        xyz : ArrayLike
            Reference point to query.
        radius : float
            Distance (in microns) used in ball point query.

        Returns
        -------
        List[int]
            Node IDs whose distance from "xyz" is less than "radius" microns.
        """
        return self.node_kdtree.query_ball_point(xyz, radius)

    def get_node_segment_id(self, node):
        """
        Gets the segment ID corresponding to the given node.

        Parameters
        ----------
        node : int
            Node ID.

        Returns
        -------
        str
            Segment ID corresponding to the given node.
        """
        return self.get_swc_name(node).split(".")[0]

    def get_swc_name(self, i):
        """
        Gets the SWC ID of the given node.

        Parameters
        ----------
        i : int
            Node ID.

        Returns
        -------
        str
            SWC ID of the given node.
        """
        component_id = self.node_component_id[i]
        return self.component_id_to_swc_name[component_id]

    def midpoint(self, i, j):
        """
        Computes the midpoint between the 3D coordinates corresponding to the
        given nodes.

        Parameters
        ----------
        i : int
            Node ID.
        j : int
            Node ID.

        Returns
        -------
        numpy.ndarray
            Midpoint between the 3D coordinates corresponding to the given
            nodes.
        """
        return np.mean([self.node_xyz[i], self.node_xyz[j]], axis=0)

    def near_leaf(self, node, radius):
        """
        Checks if the given node is within "radius" microns of a leaf.

        Parameters
        ----------
        node : int
            Node ID.
        radius : float
            Distance (in microns) to search subgraph centered at "node".

        Returns
        -------
        bool
            Indication if the given node is within "radius" microns of a leaf.
        """
        queue = [(node, 0)]
        visited = {node}
        while queue:
            # Visit node
            i, dist_i = queue.pop()
            if self.degree[i] == 1:
                return True

            # Update queue
            for j in self.neighbors(i):
                dist_j = dist_i + self.dist(i, j)
                if j not in visited and dist_j < radius:
                    queue.append((j, dist_j))
                    visited.add(j)
        return False

    def reassign_component_ids(self):
        """
        Reassigns component IDs for all connected components in the graph.
        """
        component_id_to_swc_name = dict()
        for i, nodes in enumerate(nx.connected_components(self)):
            nodes = np.array(list(nodes), dtype=int)
            component_id_to_swc_name[i + 1] = self.get_swc_name(nodes[0])
            self.node_component_id[nodes] = i + 1
        self.component_id_to_swc_name = component_id_to_swc_name

    def relabel_nodes(self):
        """
        Reassigns contiguous node IDs and update all dependent structures.
        """
        # Set node ids
        old_node_ids = np.array(self.nodes, dtype=int)
        new_node_ids = np.arange(len(old_node_ids))

        # Set edge ids
        old_to_new = dict(zip(old_node_ids, new_node_ids))
        old_edge_ids = list(self.edges)
        edge_attrs = {(i, j): data for i, j, data in self.edges(data=True)}

        # Reset graph
        self.clear()
        for (i, j) in old_edge_ids:
            self.add_edge(old_to_new[i], old_to_new[j], **edge_attrs[(i, j)])

        # Update attributes
        self.node_xyz = self.node_xyz[old_node_ids]
        self.node_component_id = self.node_component_id[old_node_ids]

        self.reassign_component_ids()
        self.node_kdtree = KDTree(self.node_xyz)
