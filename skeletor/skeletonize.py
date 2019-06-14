from .import methods

METHODS = {'teasar': methods.teasar,  'thinning': methods.thinning}
DEFAULT_PARAMS = {'teasar': {'penalty_scale': 100000,
                             'penalty_exponent': 4,
                             'mask_scale': 10,
                             'mask_min_radius': 50}}


def get_method_names():
    return list(METHODS.keys())


def get_method_params(name):
    return DEFAULT_PARAMS.get(name, {})


def skeletonize(obj, resolution=None, boundary_distances=None,
                method='teasar', **method_params):
    """ Skeletonize object defined by binary mask.

    Arguments:
        obj [np.ndarray] - binary object mask
        resolution [int, float or list] - size of the voxels in physical units,
            can be list for anisotropic input (default: None)
        boundary_distances [np.ndarray] - distance to object boundaries
            can be pre-computed for teasar (default: None)
        method [str] - method used for skeletonization (default: teasar)
        method_params [kwargs] - parameter for skeletonization method.
            For details see 'skeletor.methods'
    """
    impl = METHODS.get(method, None)
    if impl is None:
        raise KeyError("Inalid method %s, expect one of %s" % (method, str(METHODS)))
    params = DEFAULT_PARAMS.get(method, {})
    params.update(method_params)

    if resolution is None:
        resolution = [1, 1, 1]
    if isinstance(resolution, int):
        resolution = 3 * [resolution]

    nodes, edges = impl(obj, resolution, boundary_distances, **params)
    return nodes, edges
