from scipy.ndimage import convolve


def make_3d_boundaries(segmentation):
    """ Make 3d edge volume from 3d segmentation
    """
    gz = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(3, 1, 1))
    gy = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(1, 3, 1))
    gx = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(1, 1, 3))
    return ((gx ** 2 + gy ** 2 + gz ** 2) > 0)


def make_2d_boundaries(segmentation):
    """ Make 3d edge volume from 3d segmentation
        but only compute the edges in xy-plane.
        This may be more appropriate for anisotropic volumes.
    """
    gy = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(1, 3, 1))
    gx = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(1, 1, 3))
    return ((gx ** 2 + gy ** 2) > 0)
