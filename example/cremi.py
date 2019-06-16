import h5py
import numpy as np

# FIXME something is wrong with ndimage.label here
# from scipy.ndimage import label
from vigra.analysis import labelVolume

import skeletor
from cremi_tools.viewer.volumina import view


def skeletonize_largest_object(bb=np.s_[:], methods=['thinning'], prune=False):
    # cremi data, download from
    # https://cremi.org/
    path = '/g/kreshuk/data/cremi/original/sample_A_20160501.hdf'

    print("Load volume ...")
    with h5py.File(path) as f:
        seg = f['volumes/labels/neuron_ids'][bb]
        raw = f['volumes/raw'][bb]

    print("Extract largest object ...")
    # FIXME
    # seg, _ = label(seg)
    seg = labelVolume(seg.astype('uint32'))
    ids, sizes = np.unique(seg, return_counts=True)
    obj = seg == ids[np.argmax(sizes)]
    bb = np.where(obj)
    bb = tuple(slice(int(b.min()),
                     int(b.max()) + 1) for b in bb)
    obj = obj[bb]
    raw = raw[bb]

    data = [raw, obj.astype('uint32')]
    labels = ['raw', 'objects']

    # resolution of the cremi data in nm
    resolution = [40, 4, 4]
    # minimal path length used for pruning in nm
    min_path_len = 250

    for method in methods:
        print("Skeletonize with", method, "...")
        nodes, edges = skeletor.skeletonize(obj, resolution=resolution)
        vol = skeletor.nodes_to_volume(obj.shape, nodes, dilate_by=1)
        data.append(vol)
        labels.append('skeleton-%s' % method)

        if prune:
            nodes, edges = skeletor.prune(obj, nodes, edges,
                                          resolution, min_path_len)
            vol = skeletor.nodes_to_volume(obj.shape, nodes, dilate_by=1)
            data.append(vol)
            labels.append('skeleton-%s-pruned' % method)

    view(data, labels)


if __name__ == '__main__':
    bb = np.s_[:25, :256, :256]
    skeletonize_largest_object(bb, methods=['thinning'], prune=True)
