import h5py
import numpy as np
import skeletor
from scipy.ndimage import distance_transform_edt
from skeletor.utils import make_2d_boundaries
from cremi_tools.viewer.volumina import view
from scipy.ndimage.morphology import binary_dilation


def cremi_single_skeleton(bb=np.s_[:]):
    path = '/home/pape/Work/data/cremi/sample_A_20160501.hdf'
    print("Load volume ...")
    with h5py.File(path) as f:
        seg = f['volumes/labels/neuron_ids'][bb]
        raw = f['volumes/raw'][bb]

    print("Extract object ...")
    ids, sizes = np.unique(seg, return_counts=True)
    obj = seg == ids[np.argmax(sizes)]
    bb = np.where(obj)
    bb = tuple(slice(int(b.min()),
                     int(b.max()) + 1) for b in bb)
    seg = seg[bb]
    obj = obj[bb]
    raw = raw[bb]
    # view([seg.astype('uint32'), obj.astype('uint32')])

    print("Compute distance trafo ...")
    voxel_size = [10, 1, 1]
    boundaries = make_2d_boundaries(seg)
    boundary_distances = distance_transform_edt(boundaries, sampling=voxel_size)

    print("Skeletonize ...")
    d1, d2, root, src = skeletor.skeletonize(obj, boundary_distances,
                                             voxel_size=voxel_size)
    vol1 = np.zeros_like(d1, dtype='uint32')
    vol1[root[0], root[1], root[2]] = 1
    vol1 = binary_dilation(vol1, iterations=2)

    vol2 = np.zeros_like(d1, dtype='uint32')
    vol2[src[0], src[1], src[2]] = 1
    vol2 = binary_dilation(vol2, iterations=2)
    # print(skel.min(), skel.max())
    view([raw, obj.astype('uint32'), d1, d2, vol1, vol2],
         ['raw', 'obj', 'root-dist', 'initial-dist', 'root', 'src'])


if __name__ == '__main__':
    # bb = np.s_[:30, :512, :512]
    bb = np.s_[:5, :256, :256]
    cremi_single_skeleton(bb)
