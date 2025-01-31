"""
Created on Sat November 04 15:30:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Class of graphs built from swc files where each entry corresponds to a node in
the graph.

"""

from scipy.spatial import KDTree

import networkx as nx
import os

from neuron_fragment_extractor.utils import swc_util, util


class SkeletonGraph(nx.Graph):
    """
    Class of graphs built from SWC files. Each SWC file is stored as a
    distinct graph and each node in this graph.

    """

    def __init__(
        self,
        anisotropy=(1.0, 1.0, 1.0),
        img_patch_origin=None,
        img_patch_shape=None,
    ):
        """
        Constructs a DenseGraph object from a directory of SWC files.

        Parameters
        ----------
        ...

        Returns
        -------
        None

        """
        # Call parent class
        super(SkeletonGraph, self).__init__()

        # Instance attributes
        self.anisotropy = anisotropy
        self.img_bbox = util.init_bbox(img_patch_origin, img_patch_shape)
        self.swc_reader = swc_util.Reader()
        self.xyz_to_node = dict()

    # --- Constructor Helpers ---
    def load_swcs(self, swc_pointer):
        """
        Initializes graphs by reading SWC files at "swc_paths" and loading
        contents into graph structure. Note that SWC files are assumed to be
        in physical coordiantes.=

        Parameters
        ----------
        swc_pointer : Any
            Object that points to SWC files to be read, see "swc_util.py"
            documentation for details.

        Returns
        -------
        None

        """
        for swc_dict in self.swc_reader.load(swc_pointer):
            graph = swc_util.to_graph(swc_dict, set_attrs=True)
            self.clip_branches(graph)
            self.add_graph(graph)

    def clip_branches(self, graph):
        """
        Deletes all nodes from "graph" that are not contained in the bounding
        box specified by "self.img_bbox".

        Parameters
        ----------
        graph : networkx.Graph
            Graph to be searched

        Returns
        -------
        None

        """
        if self.img_bbox:
            delete_nodes = set()
            for i in graph.nodes:
                voxel = util.to_voxels(graph.nodes[i]["xyz"], self.anisotropy)
                if not util.is_contained(self.img_bbox, voxel):
                    delete_nodes.add(i)
            graph.remove_nodes_from(delete_nodes)

    def add_graph(self, graph):
        # Add nodes
        old_to_new = dict()
        for old_id, data in graph.nodes(data=True):
            new_id = self.number_of_nodes() + 1
            old_to_new[old_id] = new_id
            self.add_node(new_id, **data)
            self.xyz_to_node[tuple(data["xyz"])] = new_id

        # Add edges
        for i, j in graph.edges:
            self.add_edge(old_to_new[i], old_to_new[j])

    def init_kdtree(self):
        """
        Builds a KD-Tree from the xyz coordinates from every node stored in
        self.graphs.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.kdtree = KDTree(list(self.xyz_to_swc.keys()))

    # --- General Routines ---
    def get_projection(self, xyz):
        """
        Projects "xyz" onto "self by finding the closest point.

        Parameters
        ----------
        xyz : numpy.ndarray
            xyz coordinate to be queried.

        Returns
        -------
        numpy.ndarray
            Projection of "xyz".

        """
        _, idx = self.kdtree.query(xyz, k=1)
        return tuple(self.kdtree.data[idx])

    def save(self, output_dir):
        """
        Saves "self" to an swc file.

        Parameters
        ----------
        output_dir : str
            Path to directory that swc files are written to.

        Returns
        -------
        None

        """
        for swc_id, graph in self.graphs.items():
            cnt = 0
            for component in nx.connected_components(graph):
                entry_list = self.make_entries(graph, component)
                path = os.path.join(output_dir, f"{swc_id}.swc")
                while os.path.exists(path):
                    path = os.path.join(output_dir, f"{swc_id}.{cnt}.swc")
                    cnt += 1
                swc_util.write(path, entry_list)

    def make_entries(self, graph, component):
        """
        Makes swc entries corresponding to nodes in "component".

        Parameters
        ----------
        graph : networkx.Graph
            Graph that "component" is a connected component of.
        component : set
            Connected component of "graph".

        Returns
        -------
        entry_list
            List of swc entries generated from nodes in "component".

        """
        node_to_idx = dict()
        entry_list = []
        for i, j in nx.dfs_edges(graph.subgraph(component)):
            # Initialize
            if len(entry_list) == 0:
                node_to_idx[i] = 1
                x, y, z = tuple(graph.nodes[i]["xyz"])
                r = graph.nodes[i]["radius"]
                entry_list.append(f"1 2 {x} {y} {z} {r} -1")

            # Create entry
            node_to_idx[j] = len(entry_list) + 1
            x, y, z = tuple(graph.nodes[j]["xyz"])
            r = graph.nodes[j]["radius"]
            entry_list.append(
                f"{node_to_idx[j]} 2 {x} {y} {z} {r} {node_to_idx[i]}"
            )
        return entry_list
