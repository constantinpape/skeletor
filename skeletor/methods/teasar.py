def teasar(obj, resolution, boundary_distances=None,
           penalty_scale=100000, penalty_exponent=4,
           mask_scale=10, mask_min_radius=50):
    """
    Skeletonize object with teasar.

    Not implemented yet.

    Arguments:
        obj [np.ndarray] - binary object mask
        resolution [list] - size of the voxels in physical unit
        boundary_distances [np.ndarray] - distance to object boundaries
            can be pre-computed for teasar (default: None)
        penalty_scale [float] - scale to weight boundary distance vs.
            root distance contributrion to edge distance (default: 10000)
        penalty_exponent [int] - exponent in edge distance calculation (default: 4)
        mask_scale [float] - multiple of boundary distance used in path masking (default: 10)
        mask_min_radius [float] - minimal radius used in path masking (default: 50)
    """
    raise NotImplementedError("Teaser skeletonization is not implemented yet")

    # check if the boundary distances were pre-computed
    if boundary_distances is None:
        # TODO
        boundary_distances = ''

    # mask the boundary distances that are outside of the object
    boundary_distances = (boundary_distances * obj).astype('float32')

    # TODO
    # compute the skeleton
    nodes, edge = None, None

    return nodes, edges
