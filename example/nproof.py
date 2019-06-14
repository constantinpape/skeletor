import h5py
import numpy as np
import vigra

import skeletor
from cremi_tools.viewer.volumina import view


def fib_single_skeleton(bb=np.s_[:]):
    path = '/g/kreshuk/data/FIB25/training_blocks/raw/raw_block1.h5'
    with h5py.File(path) as f:
        raw = f['data'][bb]
    print(raw.shape)

    path = '/g/kreshuk/data/FIB25/training_blocks/gt/gt_block1.h5'
    with h5py.File(path) as f:
        seg = f['data'][bb]
    print(seg.shape)

    seg = vigra.analysis.labelVolume(seg)
    ids, sizes = np.unique(seg, return_counts=True)
    obj_id = ids[np.argmax(sizes)]
    print("Extract object", obj_id, "...")
    obj = seg == obj_id
    bb = np.where(obj)
    bb = tuple(slice(int(b.min()),
                     int(b.max()) + 1) for b in bb)
    seg = seg[bb]
    obj = obj[bb]
    raw = raw[bb]
    # view([seg.astype('uint32'), obj.astype('uint32')])

    print("Skeletonize ...")
    resolution = 8
    nodes, edges = skeletor.skeletonize(obj, resolution=resolution)

    node_coords = tuple(np.array([n[i] for n in nodes]) for i in range(3))
    vol = np.zeros_like(obj, dtype='uint32')
    vol[node_coords] = obj_id

    view([raw, obj.astype('uint32'), vol],
         ['raw', 'obj', 'skeleton'])


def vanilla_kimimaro(bb):
    from kimimaro import skeletonize
    path = '/g/kreshuk/data/FIB25/training_blocks/gt/gt_block1.h5'
    with h5py.File(path) as f:
        seg = f['data'][bb]

    mask = (seg == 363).astype('uint32')
    print(np.sum(mask))

    print("Start skeletonize ...")
    voxel_size = [8, 8, 8]
    skels = skeletonize(seg, anisotropy=voxel_size, parallel=8, fix_borders=False)
    print(type(skels))

    vol = np.zeros_like(seg, dtype='uint32')
    for seg_id, skel in skels.items():
        nodes = skel.vertices
        nodes /= np.array(voxel_size)
        nodes = nodes.astype('uint32')
        node_coords = tuple(np.array([n[i] for n in nodes]) for i in range(3))

        vol[node_coords] = seg_id

    path = '/g/kreshuk/data/FIB25/training_blocks/raw/raw_block1.h5'
    with h5py.File(path) as f:
        raw = f['data'][bb]

    view([raw, seg, vol, mask])


if __name__ == '__main__':
    bb = np.s_[:50]
    fib_single_skeleton(bb)
    # vanilla_kimimaro(bb)
