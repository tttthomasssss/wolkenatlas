import logging
import os
import pickle

import numpy as np

from wolkenatlas.util import constants


def load_pickle(filename):
    with open(filename, 'rb') as in_file:
        inv_idx = pickle.load(in_file)

    return inv_idx


def save_pickle(inv_idx, filename):
    with open(filename, 'wb') as out_file:
        pickle.dump(inv_idx, out_file)


def save_array(obj, filename):
    with open(filename, 'wb') as out_file:
        np.save(out_file, obj)


def load_array(filename):
    with open(filename, 'rb') as in_file:
        arr = np.load(in_file)

    return arr


def numpy_to_hdf(obj, filename):
    try:
        import tables
    except ImportError:
        logging.error(
            'You need to have tables installed for using this function!\nTry running `pip install tables`!')

    obj_name = os.path.splitext(os.path.basename(filename))[0]
    with tables.open_file(filename, 'w') as f:
        atom = tables.Atom.from_dtype(obj.dtype)
        arr = f.create_carray(f.root, obj_name, atom, obj.shape)
        arr[:] = obj


def hdf_to_numpy(filename, compression_level=0, compression_lib='zlib'):
    try:
        import tables
    except ImportError:
        logging.error(
            'You need to have tables installed for using this function!\nTry running `pip install tables`!')

    obj_name = os.path.splitext(os.path.basename(filename))[0]
    filters = tables.Filters(complevel=compression_level, complib=compression_lib)
    with tables.open_file(filename, 'r', filters=filters) as f:
        arr = np.array(getattr(f.root, obj_name).read())

    return arr


def load_vector_space(model_file):

    if os.path.exists(os.path.join(model_file, constants.VECTORS_FILENAME_NPY)):
        return load_array(os.path.join(model_file, constants.VECTORS_FILENAME_NPY))
    elif os.path.exists(os.path.join(model_file, constants.VECTORS_FILENAME_HDF)):
        return hdf_to_numpy(os.path.join(model_file, constants.VECTORS_FILENAME_HDF))
    else:
        raise FileNotFoundError(f'No vector file (either "X.npy" or "X.hdf") found at path "{model_file}"!')


def save_vector_space(obj, filename, use_hdf):
    if use_hdf:
        numpy_to_hdf(obj, os.path.join(filename, constants.VECTORS_FILENAME_HDF))
    else:
        save_array(obj, os.path.join(filename, constants.VECTORS_FILENAME_NPY))