#!/usr/bin/env python
import os
import sys

from pathlib import Path

import cv2
import numpy as np
import h5py as hdf5


SQUARE_SIDE_LENGTH = 227
LABELS = [
    'bb',
    'bk',
    'bn',
    'bp',
    'bq',
    'br',
    'empty',
    'wb',
    'wk',
    'wn',
    'wp',
    'wq',
    'wr'
]
OUT_PATHS = ['output_train', 'output_test']
HDF5_DATAFILE = 'data/chess_dataset.h5'
# Structure:
# build_path/
#     output_train/
#         bb/
#         bk/
#         ...
#     output_test/
#         bb/
#         bk/
#         ...

_90d_rotation_matrix = cv2.getRotationMatrix2D((SQUARE_SIDE_LENGTH / 2, SQUARE_SIDE_LENGTH / 2), 90, 1)


def create_rotations(img_path):
    img_p = Path(img_path)
    if img_p.exists():
        rotates = ['_90', '_180', '_270']
        print('  |-> Creating rotated samples on ', img_path)
        for lb in LABELS:
            p = img_p.joinpath(lb)
            img_list = p.glob('*.jpg')
            for f in img_list:
                img_cv = cv2.imread(str(f))
                for rot in rotates:
                    img_cv = cv2.warpAffine(img_cv, _90d_rotation_matrix, (SQUARE_SIDE_LENGTH, SQUARE_SIDE_LENGTH))
                    out_f = f.with_name('{}{}{}'.format(f.stem, rot, f.suffix))
                    cv2.imwrite(str(out_f), img_cv)
                    print('    |-> ', p.name, ': ', out_f.name)
    else:
        print(img_path, ' does not exist!!!')


def iter_len(iter):
    return sum(1 for _ in iter)


def samples_count(build_path):
    count = 0
    for out_path in OUT_PATHS:
        for lb in LABELS:
            out_p = build_path.joinpath(out_path, lb)
            count += iter_len(out_p.glob('*.jpg'))
    return count

def make_hdf5_dataset(build_path, n_samples):
    x_shape = (n_samples, SQUARE_SIDE_LENGTH, SQUARE_SIDE_LENGTH, 3)
    all_y = []
    print('  |-> Creating HDF5 dataset file...')
    hdf5_data_p = Path(HDF5_DATAFILE).parent
    if not hdf5_data_p.exists():
        hdf5_data_p.mkdir(parents=True)
    hdf5_f = hdf5.File(HDF5_DATAFILE, 'w')
    dset_x = hdf5_f.create_dataset('chess_imgs', x_shape, dtype='uint8')
    k_sample = 0
    for out_path in OUT_PATHS:
        for i, lb in enumerate(LABELS):
            out_p = build_path.joinpath(out_path, lb)
            img_list = out_p.glob('*.jpg')
            for f in img_list:
                img_matrix = cv2.imread(str(f))
                dset_x[k_sample] = img_matrix
                all_y.append([i, np.string_(lb)])
                k_sample += 1
    if all_y:
        hdf5_f.create_dataset('chess_labels', data=np.array(all_y))
        print('  |-> ', HDF5_DATAFILE, ' created.')
    else:
        print('  |-> ', HDF5_DATAFILE, ' NOT created! No samples found!')
    hdf5_f.close()


def main(build_path, tests_rotate=False):
    build_p = Path(build_path)
    if build_p.exists():
        print(' -> Imageset path: ', str(build_p))
        if tests_rotate:
            # Make rotated test samples
            create_rotations(str(build_p.joinpath('output_test')))
        n_samples = samples_count(build_p)
        print(' |-> Total samples: ', n_samples)
        make_hdf5_dataset(build_p, n_samples)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        with_rotate = False
        if len(sys.argv) > 2:
            with_rotate = sys.argv[2] == '--with-tests-rotate'
        main(sys.argv[1], with_rotate)
    else:
        print('\nUsage: {} path/to/image/dataset [--with-tests-rotate]\n'.format(sys.argv[0]))
        print('  NOTE: We assume that there are the folders:')
        print('         - output_train')
        print('         - output_test')
        print('        inside path/to/image/dataset\n')
