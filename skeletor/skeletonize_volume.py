import numpy as np
from .skeletonize import skeletonize


def apply_mask(segmentation, object_ids, in_place):
    if not in_place:
        segmentation = np.copy(segmentation)
    segmentation[np.logical_not(np.isin(segmentation, object_ids))] = 0
    return segmentation


# TODO parameter
def skeletonize_dense(segmentation, voxel_size=None, object_ids=None,
                      in_place=True, min_seg_size=1000, **teasar_parameter):
    """ Skeletonize all objects in segmentation volume.
    Note that we assume that all objects in the segmentation are connected.

    Arguments:
        segmentation [np.ndarray] - volumetric segmentation
        voxel_size [int, float or list] - size of the voxels,
            can be list for anisotropic input (default: None)
        object_ids [list] - list of object ids to skeletonize.
            By default (None), all objects are skeletonized
        in_place [bool] - do we allow changes to the segmentation array (default: True)
        min_seg_size [int] - minimal segment size for skeletonization (default: 1000)
    Keyword Arguments:
        teasar_parameter - parameter for the teasar skeletonization:
    """
    raise NotImplementedError("TODO")

    if object_ids is not None:
        segmentation = apply_mask(segmentation, object_ids, in_place)

    # filter small components
    seg_ids, seg_sizes = np.unique(segmentation, return_counts=True)
    seg_ids = [seg_id for seg_id, seg_size in zip(seg_ids, seg_sizes)
               if seg_size > min_seg_size]
    if seg_ids[0] == 0:
        seg_ids = seg_ids[1:]

    # make boundaries and compute distances
    # TODO use 2d or 3d boundaries ?
    boundary_distances = utils.make_2d_boundaries(segmentation)
    boundary_distances = distance_transform_edt(boundary_distances, sampling=voxel_size)

    # skeletonize the objects in the segmentation
    slices = find_objects(segmentation)
    skeletons = {}
    for obj_id in seg_ids:
        slice_ = slices[obj_id - 1]
        if slice_ is None:
            continue

        # crop object mask and distances to segment's roi
        obj = segmentation[slice_]
        obj = obj == obj_id
        inner_distances = boundary_distances[slice_]
        skeleton = skeletonize(obj, inner_distances, **teasar_parameter)

        # TODO offset the skeleton ids with the bb offset
        # offset = [sl.start for sl in slice_]

        skeletons[obj_id] = skeleton

    return skeletons
