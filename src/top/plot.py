from typing import Optional

import networkx as nx
import matplotlib.pyplot as plt

from top.problem import Solution


def plot_solution(solution: Solution, ax: Optional[plt.Axes] = None, node_size: int = 300, show: bool = True):
    """Plot the given `solution`.

    - Plots only edges (legs) that are part of the routes in the solution.
    - Nodes are positioned using the customers' (x, y) coordinates.
    - Node label is the customer's reward.

    Returns the matplotlib Axes containing the plot.
    """
    problem = solution.problem
    customers = problem.customers
    if not customers:
        raise ValueError("Problem has no customers to plot")

    # Positions for nodes: use (x, y)
    pos = {cid: (c.x, c.y) for cid, c in customers.items()}

    # Nodes to include: all customers (colors set per id)
    all_nodes = list(customers.keys())
    max_id = max(all_nodes)

    node_colors = []
    for cid in all_nodes:
        if cid == 0:
            node_colors.append("green")
        elif cid == max_id:
            node_colors.append("red")
        else:
            node_colors.append("blue")

    # Collect edges only for legs covered by the routes (consecutive customers in each route)
    edges = []
    for route in solution.routes.values():
        custs = route.customers
        # add directed or undirected legs between consecutive customers
        for a, b in zip(custs, custs[1:]):
            edges.append((a, b))

    G = nx.DiGraph()
    G.add_nodes_from(all_nodes)
    if edges:
        G.add_edges_from(edges)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True

    nx.draw_networkx_nodes(G, pos, nodelist=all_nodes, node_color=node_colors, node_size=node_size, ax=ax)
    if edges:
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color="black", arrows=True, ax=ax)

    # Labels: use reward for each customer
    labels = {cid: str(customers[cid].reward) for cid in all_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)

    ax.set_aspect("equal")
    ax.set_axis_off()

    if show:
        plt.tight_layout()
        plt.show()

    return ax
