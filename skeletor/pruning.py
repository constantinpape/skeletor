import numpy as np
import nifty
from .util import edges_to_graph, simplify_skeleton


def compute_terminals(edges):
    skel_graph, n_nodes = edges_to_graph(edges)
    degrees = np.array([len([adj for adj in skel_graph.nodeAdjacency(u)])
                        for u in range(n_nodes)])
    terminals = np.where(degrees == 1)[0]
    return terminals


# TODO vectorize
# TODO This only works if the skeleton has same connectivity
# as the grid graph (= direct nhood)
def compute_runlength(path, nodes, grid_graph, weights):
    runlen = 0.
    for p1, p2 in zip(path[:-1], path[1:]):
        coord1, coord2 = nodes[p1].tolist(), nodes[p2].tolist()
        u, v = grid_graph.coordinateToNode(coord1), grid_graph.coordinateToNode(coord2)
        edge = grid_graph.findEdge(u, v)
        runlen += weights[edge]
    return runlen


def prune_paths(nodes, edges, paths, to_prune):
    # NOTE the first node in the path is always a junction, so we don't prune it
    prune_nodes = (np.array([pnode for pnodes in paths for pnode in pnodes[1:]]),)
    nodes = nodes[prune_nodes]
    edge_filter = np.isin(edges, prune_nodes[0]).any(axis=1)
    edges = edges[edge_filter]
    return nodes, edges


def prune_short_paths(nodes, edges,
                      grid_graph, weights,
                      terminals, min_path_len):
    # compute paths in the skeleton
    paths = simplify_skeleton(edges)

    # filter for paths involving a terminal
    # note the first node is always a junction, so only the last can be a terminal
    paths = [path for path in paths if path[-1] in terminals]

    # compute the run-len for all paths
    run_lengths = [compute_runlength(path, nodes, grid_graph, weights)
                   for path in paths]
    to_prune = [path_id for path_id, rlen in enumerate(run_lengths) if rlen < min_path_len]

    nodes, edges = prune_paths(nodes, edges, paths, to_prune)
    return nodes, edges


def prune_by_clustering(nodes, edges, grid_graph, weights, terminals):

    # need to cast to graph in order to use nifty dijkstra impl
    graph = nifty.graph.undirectedGraph(grid_graph.numberOfNodes)
    graph.insertEdges(grid_graph.uvIds())
    dijkstra = nifty.graph.ShortestPathDijkstra(graph)

    # compute all inter terminal distances
    n_terminals = len(terminals)
    distances = np.zeros((n_terminals, n_terminals), dtype='float32')
    for t in terminals:
        source = grid_graph.coordinateToNode(nodes[t].tolist())
        targets = terminals[terminals > t]
        targets = [grid_graph.coordinateToNode(nodes[target].tolist()) for target in targets]
        shortest_paths = dijkstra.runSingleSourceMultiTarget(weights, source, targets,
                                                             returnNodes=False)
        path_distances = np.array([np.sum(weights[path]) for path in shortest_paths])
        distances[t, t:] = path_distances

    # TODO
    # cluster terminals w.r.t distances

    # TODO
    # drop all terminal paths that do not correspond to a cluster center

    return nodes, edges


# TODO params for clustering based pruning?
def prune(obj, nodes, edges, resolution,
          min_path_len=None, by_clustering=False):

    # compute the grid graph and the distances euclidean distances
    shape = obj.shape
    grid_graph = nifty.graph.undirectedGridGraph(shape)
    weights = grid_graph.euclideanEdgeMap(obj, resolution)

    # find terminals
    terminals = compute_terminals(edges)

    # drop short paths
    if min_path_len is not None:
        nodes, edges = prune_short_paths(nodes, edges,
                                         grid_graph, weights,
                                         terminals, min_path_len)
        # update terminals
        terminals = compute_terminals(edges)

    if by_clustering:
        nodes, edges = prune_by_clustering(nodes, edges,
                                           grid_graph, weights,
                                           terminals)

    return nodes, edges
