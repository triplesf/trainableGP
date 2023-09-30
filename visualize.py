""" Network architecture visualizer using graphviz """
import pygraphviz as pgv
import gp_tree
import os


def draw_graph(expr, save_dir, save_name):
    nodes, edges, labels = graph(expr)

    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw(os.path.join(save_dir, save_name))


def graph(expr):
    """Construct the graph of a tree expression. The tree expression must be
    valid. It returns in order a node list, an edge list, and a dictionary of
    the per node labels. The node are represented by numbers, the edges are
    tuples connecting two nodes (number), and the labels are values of a
    dictionary for which keys are the node numbers.

    :param expr: A tree expression to convert into a graph.
    :returns: A node list, an edge list, and a dictionary of labels.

    The returned objects can be used directly to populate a
    `pygraphviz <http://networkx.lanl.gov/pygraphviz/>`_ graph::

        import pygraphviz as pgv

        # [...] Execution of code that produce a tree expression

        nodes, edges, labels = graph(expr)

        g = pgv.AGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        g.layout(prog="dot")

        for i in nodes:
            n = g.get_node(i)
            n.attr["label"] = labels[i]

        g.draw("tree.pdf")

    or a `NetworX <http://networkx.github.com/>`_ graph::

        import matplotlib.pyplot as plt
        import networkx as nx

        # [...] Execution of code that produce a tree expression

        nodes, edges, labels = graph(expr)

        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        pos = nx.graphviz_layout(g, prog="dot")

        nx.draw_networkx_nodes(g, pos)
        nx.draw_networkx_edges(g, pos)
        nx.draw_networkx_labels(g, pos, labels)
        plt.show()


    .. note::

       We encourage you to use `pygraphviz
       <http://networkx.lanl.gov/pygraphviz/>`_ as the nodes might be plotted
       out of order when using `NetworX <http://networkx.github.com/>`_.
    """
    nodes = list(range(len(expr)))
    edges = list()
    labels = dict()

    stack = []
    for i, node in enumerate(expr):
        if stack:
            edges.append((stack[-1][0], i))
            stack[-1][1] -= 1
        labels[i] = node.name if isinstance(node, gp_tree.Primitive) else node.value
        stack.append([i, node.arity])
        while stack and stack[-1][1] == 0:
            stack.pop()

    return nodes, edges, labels
