import numpy as np
from skan import csr

#
# Parser for custom n5 skeleton format
# TODO
# get rid of skan
# adjust to in-memory skeleton format returned by `skeletonize`
#


def read_n5(ds, skel_id):
    """ Read skeleton stored in custom n5-based format

    The skeleton data is stored via varlen chunks: each chunk contains
    the data for one skeleton and stores:
    (n_skel_points, coord_z_0, coord_y_0, coord_x_0, ..., coord_z_n, coord_y_n, coord_x_n,
     n_edges, edge_0_u, edge_0_v, ..., edge_n_u, edge_n_v)

    Arguments:
        ds [z5py.Dataset]: input dataset
        skel_id [int]: id of the object corresponding to the skeleton
    """
    # read data from chunk
    data = ds.read_chunk((skel_id,))

    # check if the chunk is empty
    if data is None:
        return None, None

    # read number of points and coordinates
    n_points = data[0]
    offset = 1
    coord_len = int(3 * n_points)
    coords = data[offset:offset+coord_len].reshape((n_points, 3))
    offset += coord_len
    # read number of edges and edges
    n_edges = data[offset]
    offset += 1
    edge_len = int(2 * n_edges)
    assert len(data) == offset + edge_len, "%i, %i" % (len(data), offset + edge_len)
    edges = data[offset:offset+edge_len].reshape((n_edges, 2))
    return coords, edges


def write_n5(ds, skel_id, skel_vol, coordinate_offset=None):
    """ Write skeleton to custom n5-based format

    The skeleton data is stored via varlen chunks: each chunk contains
    the data for one skeleton and stores:
    [n_skel_points, coord_z_0, coord_y_0, coord_x_0, ..., coord_z_n, coord_y_n, coord_x_n,
     n_edges, edge_0_u, edge_0_v, ..., edge_n_u, edge_n_v]

    Arguments:
        ds [z5py.Dataset]: output dataset
        skel_id [int]: id of the object corresponding to the skeleton
        skel_vol [np.ndarray]: binary volume containing the skeleton
        coordinate_offset [listlike]: offset to coordinate (default: None)
    """
    # NOTE looks like skan function names are about to change in 0.8:
    # csr.numba_csgraph -> csr.csr_to_nbgraph
    # extract the skeleton graph

    # this may fail for small skeletons with a value-error
    try:
        pix_graph, coords, _ = csr.skeleton_to_csgraph(skel_vol)
    except ValueError:
        return
    graph = csr.numba_csgraph(pix_graph)

    # skan-indexing is 1 based, so we need to get rid of first coordinate row
    coords = coords[1:]
    # check if we have offset and add up if we do
    if coordinate_offset is not None:
        assert len(coordinate_offset) == 3
        coords += coordinate_offset

    # make serialization for number of points and coordinates
    n_points = coords.shape[0]
    data = [np.array([n_points]), coords.flatten()]

    # make edges
    edges = [[u, v] for u in range(1, n_points + 1) for v in graph.neighbors(u) if u < v]
    edges = np.array(edges)
    # substract 1 to change to zero-based indexing
    edges -= 1
    # add number of edges and edges to the serialization
    n_edges = len(edges)
    data.extend([np.array([n_edges]), edges.flatten()])

    data = np.concatenate(data, axis=0)
    ds.write_chunk((skel_id,), data.astype('uint64'), True)
