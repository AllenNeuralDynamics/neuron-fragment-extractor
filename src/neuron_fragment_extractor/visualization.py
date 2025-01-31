"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for visualizing SkeletonGraphs.

"""

import networkx as nx
import plotly.graph_objects as go


def visualize(graph):
    """
    Visualizes the given graph using Plotly.

    Parameters
    ----------
    graph : SkeletonGraph
        Graph to be visualized.

    Returns
    -------
    None

    """
    # Initializations
    data = get_edge_trace(graph)
    layout = get_layout()

    # Generate plot
    fig = go.Figure(data=data, layout=layout)
    fig.show()


def visualize_with_groundtruth(fragments_graph, groundtruth_graph):
    """
    Visualizes two graphs on the same plot using Plotly.

    Parameters
    ----------
    fragments_graph : SkeletonGraph
        Graph that representing neuron fragments to be visualized.
    groundtruth_graph : SkeletonGraph
        Graph object representing the ground truth to be visualized.

    Returns
    -------
    None

    """
    # Initializations
    fragments_data = get_edge_trace(
        fragments_graph, color="crimson", name="Fragments"
    )
    groundtruth_data = get_edge_trace(
        groundtruth_graph, color="black", name="Ground Truth"
    )
    data = [fragments_data, groundtruth_data]
    layout = get_layout()

    # Generate plot
    fig = go.Figure(data=data, layout=layout)
    fig.show()


def get_edge_trace(graph, color="black", name=""):
    """
    Generates a 3D edge trace for visualizing the edges of a graph.

    Parameters
    ----------
    graph : SkeletonGraph
        Graph to be visualized.
    color : str, optional
        Color to use for the edge lines in the plot. The default is "black".
    name : str, optional
        Name of the edge trace. The default is an empty string.

    Returns
    -------
    plotly.graph_objects.Scatter3d
        Scatter3d object that represents the 3D trace of the graph edges.

    """
    x, y, z = get_edge_xyz(graph)
    edge_trace = go.Scatter3d(
        x=x, y=y, z=z, mode="lines", line=dict(color=color, width=5), name=name
    )
    return edge_trace


# --- helpers ---
def get_edge_xyz(graph):
    """
    Extracts the 3D coordinates of the edges in the given graph.

    Parameters
    ----------
    graph : SkeletonGraph
        Graph containing node-level attribute called "xyz".

    Returns
    -------
    Tuple[float]
        A tuple containing the xyz-coordiantes for the edges.

    """
    x, y, z = list(), list(), list()
    xyz = nx.get_node_attributes(graph, "xyz")
    for u, v in graph.edges():
        x0, y0, z0 = xyz[u]
        x1, y1, z1 = xyz[v]
        x.extend([x0, x1, None])
        y.extend([y0, y1, None])
        z.extend([z0, z1, None])
    return x, y, z


def get_layout():
    """
    Generates the layout for a 3D plot using Plotly.

    Parameters
    ----------
    None

    Returns
    -------
    plotly.graph_objects.Layout
        Layout object that defines the appearance and properties of the plot.

    """
    layout = go.Layout(
        scene=dict(aspectmode="manual", aspectratio=dict(x=1, y=1, z=1)),
        showlegend=True,
        template="plotly_white",
        height=700,
        width=1200,
    )
    return layout
