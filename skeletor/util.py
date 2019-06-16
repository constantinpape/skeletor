import numpy as np
import nifty
from scipy.ndimage.morphology import binary_dilation

#
# some of these functions might be slow in pure python
# could put it into C++ (in nifty?)
#


def edges_to_graph(edges):
    n_nodes = int(edges.max()) + 1
    graph = nifty.graph.undirectedGraph(n_nodes)
    graph.insertEdges(edges)
    return graph, n_nodes


def simplify_skeleton(edges):
    """ Simplify skeleton edges to paths from terminal nodes to junctions/
    from junctions to junctions.
    """

    graph, n_nodes = edges_to_graph(edges)

    degrees = [len([adj for adj in graph.nodeAdjacency(u)])
               for u in range(n_nodes)]
    visited = np.zeros(n_nodes, dtype='bool')

    # node queue to  build paths

    queue = []
    # start with first junction and put all its neighbors on the queue
    u = np.where(degrees > 2)[0][0]
    visited[u] = 1
    for adj in graph.nodeAdjacency(u):
        queue.append((adj[0], [u]))

    paths = []
    while queue:
        # get current node
        u, path = queue.pop()
        if visited[u]:
            continue
        visited[u] = 1
        degree = degrees[u]
        path.append(u)

        # check what kind of node and take appropriate actions:
        # terminal node -> end current path
        if degree == 1:
            paths.append(path)
        # intermediate path node -> put next node on the queue
        elif degree == 2:
            # iterate over ngbs
            for adj in graph.nodeAdjacency(u):
                v = adj[0]
                if visited[v]:
                    continue
                queue.append((v, path))
        # junction node -> end current path and put next nodes on the queue
        else:
            paths.append(path)
            # iterate over ngbs
            for adj in graph.nodeAdjacency(u):
                v = adj[0]
                if visited[v]:
                    continue
                queue.append((v, [u]))
    return paths


def dfs(graph, node, parent, visited_nodes):
    """ Depth-first search starting from node to check whether it
        can be reached via cycle in graph.
    """
    visited_nodes.append(node)
    # iterate over all neighbors of the current node
    for adj in graph.nodeAdjacency(node):
        ngb = adj[0]
        # if this is the parent of the current node, continue
        if ngb == parent:
            continue
        # if this node was already visited, we have found a cycle
        if ngb in visited_nodes:
            return True
        # perform dfs search starting from ngb node
        if dfs(graph, ngb, node, visited_nodes):
            return True


def has_cycle(edges):
    """ Check if graph defined by given skeleton edges has a cycle
    """
    graph, n_nodes = edges_to_graph(edges)

    visited_nodes = []
    for node in range(n_nodes):
        if node in visited_nodes:
            continue
        if dfs(graph, node, -1, visited_nodes):
            return True
    return False


# TODO
def is_connected(edges):
    """ Check if graph defined by given skeleton edges is connected
    """
    pass


def nodes_to_volume(shape, nodes, dilate_by=0, dtype='uint32'):
    vol = np.zeros(shape, dtype=dtype)
    node_coords = tuple(np.array([n[i] for n in nodes]) for i in range(3))
    vol[node_coords] = 1
    if dilate_by > 0:
        vol = binary_dilation(vol, iterations=dilate_by).astype(dtype)
    return vol
