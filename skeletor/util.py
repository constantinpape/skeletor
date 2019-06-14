import numpy as np
import nifty


# TODO implement in C++ if this becomes a bottleneck
def simplify_skeleton(edges):
    n_nodes = int(edges.max()) + 1
    graph = nifty.graph.undirectedGraph(n_nodes)
    graph.insertEdges(edges)

    degrees = [len([adj for adj in graph.nodeAdjacency(u)])
               for u in range(n_nodes)]
    visited = np.zeros(n_nodes, dtype='bool')

    # start with a junction
    u = np.where(degrees > 2)[0][0]
    queue = [(u, u, [])]

    paths = []
    path_nodes = []
    while queue:
        # get current node
        u, start_node, this_nodes = queue.pop()
        if visited[u]:
            continue
        visited[u] = 1
        degree = degrees[u]
        this_nodes.append(u)

        # check what kind of node and take appropriate actions:
        # terminal node -> end current path
        if degree == 1:
            paths.append([start_node, u])
            path_nodes.append(np.array(this_nodes))
        # intermediate path node -> put next node on the queue
        elif degree == 2:
            visited[u] = 2  # set node to visited
            # iterate over ngbs
            for adj in graph.nodeAdjacency(u):
                # TODO
                v = adj[0]
                if visited[v]:
                    continue
                queue.append((v, start_node, this_nodes))
        # junction node -> end current path and put next nodes on the queue
        else:
            paths.append([start_node, u])
            path_nodes.append(np.array(this_nodes))
            # iterate over ngbs
            for adj in graph.nodeAdjacency(u):
                # TODO
                v = adj[0]
                if visited[v]:
                    continue
                queue.append((v, u, []))

    paths = np.array(paths, dtype='uint64')
    return paths, path_nodes
