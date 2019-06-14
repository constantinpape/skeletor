import h5py
import numpy as np
import skeletor
import edt
import vigra
from cremi_tools.viewer.volumina import view
from scipy.ndimage.morphology import binary_dilation


def cremi_single_skeleton(bb=np.s_[:]):
    # path = '/home/pape/Work/data/cremi/sample_A_20160501.hdf'
    path = '/g/kreshuk/data/cremi/original/sample_A_20160501.hdf'
    print("Load volume ...")
    with h5py.File(path) as f:
        seg = f['volumes/labels/neuron_ids'][bb]
        raw = f['volumes/raw'][bb]

    print("Extract object ...")
    seg = vigra.analysis.labelVolume(seg.astype('uint32'))
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
    voxel_size = [40, 4, 4]
    boundary_distances = edt.edt(seg, anisotropy=voxel_size,
                                 black_border=False, order='C',
                                 parallel=1)
    boundary_distances = obj * boundary_distances

    print("Skeletonize ...")
    nodes, edges = skeletor.skeletonize(obj, boundary_distances, voxel_size=voxel_size)

    print(len(nodes))
    print(len(edges))
    print(edges.max())

    node_coords = tuple(np.array([n[i] for n in nodes]) for i in range(3))

    vol = np.zeros_like(obj, dtype='uint32')
    vol[node_coords] = 1
    # vol = binary_dilation(vol, iterations=2)

    view([raw, obj.astype('uint32'), vol],
         ['raw', 'obj', 'skeleton'])


if __name__ == '__main__':
    # bb = np.s_[:30, :512, :512]
    # bb = np.s_[:5, :256, :256]
    bb = np.s_[:50]
    cremi_single_skeleton(bb)
