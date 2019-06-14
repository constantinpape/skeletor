import edt
from kimimaro.trace import trace


# TODO kmimario trace has some more parameters, mainly for soma detection
# for now, I leave it at the defaults, but would be nice to enable setting this
def teasar(obj, resolution, boundary_distances=None,
           penalty_scale=100000, penalty_exponent=4,
           mask_scale=10, mask_min_radius=50):
    """
    Skeletonize object with teasar.

    Wrapper around implementation from https://github.com/seung-lab/kimimaro.

    Arguments:
        obj [np.ndarray] - binary object mask
        resolution [int, float or list] - size of the voxels in physical units,
            can be list for anisotropic input (default: None)
        boundary_distances [np.ndarray] - distance to object boundaries
            can be pre-computed for teasar (default: None)
        penalty_scale [float] - scale to weight boundary distance vs.
            root distance contributrion to edge distance (default: 10000)
        penalty_exponent [int] - exponent in edge distance calculation (default: 4)
        mask_scale [float] - multiple of boundary distance used in path masking (default: 10)
        mask_min_radius [float] - minimal radius used in path masking (default: 50)
    """
    # check if the boundary distances were pre-computed
    if boundary_distances is None:
        boundary_distances = edt.edt(obj, anisotropy=resolution,
                                     black_border=False, order='C', parallel=1)

    # mask the boundary distances that are outside of the object
    boundary_distances = (boundary_distances * obj).astype('float32')

    # compute the skeleton
    skel = trace(obj, boundary_distances, scale=mask_scale, const=mask_min_radius,
                 anisotropy=resolution, pdrf_scale=penalty_scale, pdrf_exponent=penalty_exponent)

    # return the node coordinate list and the edges from the skeleton
    nodes = skel.vertices
    return nodes.astype('uint64'), skel.edges
