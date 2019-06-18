from concurrent import futures
import numpy as np
from scipy.ndimage import find_objects
from .skeletonize import skeletonize, get_method_names, get_method_params


def compute_boundary_distances(segmentation, resolution, n_threads):
    raise NotImplementedError("Boundary distance transform calculation is not implemented")


def apply_mask(segmentation, object_ids, in_place):
    if not in_place:
        segmentation = np.copy(segmentation)
    segmentation[np.logical_not(np.isin(segmentation, object_ids))] = 0
    return segmentation


def skeletonize_volume(segmentation, resolution=None, object_ids=None,
                       in_place=True, min_seg_size=1000,
                       method='thinning', n_threads=1, **method_params):
    """ Skeletonize objects in segmentation volume.

    Note that we assume that all objects in the segmentation are connected!

    Arguments:
        segmentation [np.ndarray] - volumetric segmentation
        resolution [int, float or list] - size of the voxels,
            can be list for anisotropic input (default: None)
        object_ids [list] - list of object ids to skeletonize.
            By default (None), all objects are skeletonized
        in_place [bool] - do we allow changes to the segmentation array (default: True)
        min_seg_size [int] - minimal segment size for skeletonization (default: 1000)
        method [str] - method used for skeletonization (default: 'thinning')
        n_threads [int] - number of threads (default: 1)
    Keyword Arguments:
        method_params - parameter for the skeletonization method
    """

    # get the skeletonization method and its parameters
    methods = get_method_names()
    if method not in methods:
        raise KeyError("Inalid method %s, expect one of %s" % (method, str(methods)))
    params = get_method_params(method)
    params.update(method_params)

    # zero out objects that were not selected
    if object_ids is not None:
        segmentation = apply_mask(segmentation, object_ids, in_place)

    # filter small objects
    seg_ids, seg_sizes = np.unique(segmentation, return_counts=True)
    seg_ids = [seg_id for seg_id, seg_size in zip(seg_ids, seg_sizes)
               if seg_size > min_seg_size]
    if seg_ids[0] == 0:
        seg_ids = seg_ids[1:]

    # if we use teasar: pre-compute boundary distances
    if method == 'teasar':
        boundary_distances = compute_boundary_distances(segmentation, resolution, n_threads)
    else:
        boundary_distances = None

    # skeletonize the objects in the segmentation
    slices = find_objects(segmentation)

    def _skeletonize(seg_id):
        slice_ = slices[seg_id - 1]

        if slice_ is None:
            return None

        # crop object mask and distances to segment's roi
        obj = segmentation[slice_]
        obj = obj == seg_id
        inner_distances = None if boundary_distances is None else boundary_distances[slice_]

        nodes, edges = skeletonize(obj, resolution=resolution,
                                   boundary_distances=inner_distances, **params)

        # offset the skeleton coordinates with the bb offset
        offset = np.array([sl.start for sl in slice_])
        nodes += offset

        return nodes, edges

    # skeletonize the objects in parallel
    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(_skeletonize, seg_id) for seg_id in seg_ids]
        skeletons = [t.result() for t in tasks]

    skeletons = {seg_id: skel for seg_id, skel in zip(seg_ids, skeletons)
                 if skel is not None}
    return skeletons
