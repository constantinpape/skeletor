from skimage.morphology import skeletonize_3d
# TODO
# from skan import ...


def thinning(obj, *args, **kwargs):
    """
    Skeletonize object with teasar.

    Wrapper around implementation from https://github.com/seung-lab/kimimaro.

    Arguments:
        obj [np.ndarray] - binary object mask
    """
    raise NotImplementedError("TODO")
    vol = skeletonize_3d(obj)

    # TODO use skan or custom functionality to extract nodes and edges

    return None, None
