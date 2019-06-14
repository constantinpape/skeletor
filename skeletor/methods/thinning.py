import numpy as np
from skimage.morphology import skeletonize_3d
# TODO check if skan does the job, otherwise need to reimplement skeleton -> graph
from skan import csr


def thinning(obj, resolution, *args, **kwargs):
    """
    Skeletonize object with teasar.

    Wrapper around implementation from https://github.com/seung-lab/kimimaro.

    Arguments:
        obj [np.ndarray] - binary object mask
        resolution [list] - size of the voxels in physical unit
    """

    # skeletonize with thinning
    vol = skeletonize_3d(obj)

    # use skan to extact skeleon node coordinates and edges
    adj_mat, nodes, _ = csr.skeleton_to_csgraph(vol, spacing=resolution)
    graph = csr.csr_to_nbgraph(adj_mat)

    # I think we need to substract 1 here, beacuse skan uses 1-based indexing
    n_nodes = len(nodes)
    edges = np.array([[u - 1, v - 1] for u in range(1, n_nodes + 1) for v in graph.neighbors(u)
                      if u < v], dtype='uint64')

    # retunr node coordinate list and edges
    return nodes.astype('uint64'), edges
