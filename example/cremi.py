import h5py
import numpy as np
import vigra

import skeletor
from cremi_tools.viewer.volumina import view


def cremi_single_skeleton(bb=np.s_[:]):
    path = '/g/kreshuk/data/cremi/original/sample_A_20160501.hdf'
    print("Load volume ...")
    with h5py.File(path) as f:
        seg = f['volumes/labels/neuron_ids'][bb]
        raw = f['volumes/raw'][bb]

    print("Extract largest object ...")
    seg = vigra.analysis.labelVolume(seg.astype('uint32'))
    ids, sizes = np.unique(seg, return_counts=True)
    obj = seg == ids[np.argmax(sizes)]
    bb = np.where(obj)
    bb = tuple(slice(int(b.min()),
                     int(b.max()) + 1) for b in bb)
    seg = seg[bb]
    obj = obj[bb]
    raw = raw[bb]

    print("Skeletonize with teasar ...")
    resolution = [40, 4, 4]
    nodes, edges = skeletor.skeletonize(obj, resolution=resolution)

    node_coords = tuple(np.array([n[i] for n in nodes]) for i in range(3))
    vol = np.zeros_like(obj, dtype='uint32')
    vol[node_coords] = 1

    view([raw, obj.astype('uint32'), vol],
         ['raw', 'obj', 'skeleton'])


if __name__ == '__main__':
    bb = np.s_[:50]
    cremi_single_skeleton(bb)
