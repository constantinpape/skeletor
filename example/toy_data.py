import numpy as np
import nifty.skeletons as nskel
from cremi_tools.viewer.volumina import view


def toy_euclid_dist_small():
    x = np.zeros((3, 10, 10), dtype='bool')
    x[:, 2:8, 2:8] = 1

    voxel_size = [1, 1, 1]
    root = [1, 5, 5]
    dist = nskel.euclidean_distance(x, root, voxel_size)

    view([x.astype('uint8'), dist])


def toy_euclid_dist_big():
    x = np.zeros((10, 100, 100), dtype='bool')
    x[2:8, 20:80, 20:80] = 1

    voxel_size = [10, 1, 1]
    root = [5, 50, 50]
    dist = nskel.euclidean_distance(x, root, voxel_size)

    view([x.astype('uint8'), dist])



if __name__ == '__main__':
    # toy_euclid_dist_small()
    toy_euclid_dist_big()
