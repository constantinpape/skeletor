import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import find_objects
import nifty.skeletons as nskel
from . import utils


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


def skeletonize(obj, boundary_distances, voxel_size=None,
                penalty_scale=5000., penalty_exponent=16,
                mask_scale=10., mask_min_radius=10.):
    """ Skeletonize segmentation object with TEASAR.

    Arguments:
        obj [np.ndarray] - binary object mask
        boundary_distances [np.ndarray] - distance to object boundaries
        voxel_size [int, float or list] - size of the voxels,
            can be list for anisotropic input (default: None)
        penalty_scale [float] - scale to weight boundary distance vs.
            root distance contributrion to edge distance (default: 5000)
        penalty_exponent [int] - exponent in edge distance calculation (default: 16)
        mask_scale [float] - multiple of boundary distance used in path masking (default: 10)
        mask_min_radius [float] - minimal radius used in path masking (default: 10)
    """

    if voxel_size is None:
        voxel_size = [1, 1, 1]

    # compute root node (= node most distance from some boundary node)
    root = find_root(obj, voxel_size)

    # compute distance fields for the edge distance field:
    # distances to root voxel
    root_distances = nskel.euclidean_distance(obj, root, voxel_size)

    # compute the penalized edge distance map
    edge_distances = compute_edge_distances(boundary_distances, root_distances,
                                            penalty_scale, penalty_exponent)
    # set distances outside of the object to inf
    edge_distances[np.logical_not(obj)] = np.inf

    # compute all skeleton paths
    print("Compute paths")
    skel_paths = compute_paths(boundary_distances, root_distances, edge_distances,
                               obj, root, voxel_size, mask_scale, mask_min_radius)

    # TODO extract actual skeleton
    return skel_paths


# TODO introduce pruning / early stopping !
def compute_paths(boundary_distances, root_distances, edge_distances, obj, root,
                  voxel_size, mask_scale, mask_min_radius):
    """ Compute the skeleton paths
    """

    valid_labels = np.count_nonzero(obj)
    # TODO instead of appending, keep list of unique coordinates in paths
    paths = []
    # keep extracting labels until all piels are explained
    # by a skeleton
    while valid_labels > 0:

        # find the next target and compute the path to it
        target = np.unravel_index(np.argmax(root_distances), root_distances.shape)
        print("Dijsktra to", target)
        path = nskel.dijkstra(edge_distances, root, list(target))
        print("done")

        # mask all pixels that are explained by this path
        # TODO path contains coordinates that are already part of prev. path
        # remove them before computing the path mask
        print("Path mask")
        path_mask = nskel.compute_path_mask(boundary_distances, path,
                                            mask_scale, mask_min_radius, voxel_size)
        print("done")
        path_mask = path_mask.reshape(obj.shape)
        obj[path_mask] = 0
        root_distances[path_mask] = 0.

        # set distances along the path to zero
        # FIXME the path gets out of range oO
        edge_distances[path] = 0.

        valid_labels -= path_mask.sum()
        paths.append(path)

    return path


def find_root(obj, voxel_size, return_dist=False):
    """ Find a root node for teasar.

    The root node can be ANY node maximally
    distant from some other boundary node.
    """
    # find any voxel on the boundary
    source_vox = nskel.boundary_voxel(obj)
    # compute the distance to this voxel and find the furthest voxel (= root)
    distance = nskel.euclidean_distance(obj, source_vox, voxel_size)
    root = np.unravel_index(np.argmax(distance), distance.shape)
    if return_dist:
        return list(root), distance
    else:
        return list(root)


def compute_edge_distances(boundary_distances, root_distances,
                           penalty_scale, penalty_exponent):
    """ Compute the penalized edge distances.
    """

    # compute the boundary distance contribution
    bd_max = boundary_distances.max() ** 1.01
    edge_distances = (1. - boundary_distances / bd_max) ** penalty_exponent

    # weight by the scale
    edge_distances *= penalty_scale

    # add the distance from root contribution
    edge_distances += (root_distances / root_distances.max())
    return edge_distances
