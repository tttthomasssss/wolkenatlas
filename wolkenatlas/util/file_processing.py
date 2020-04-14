import os
import pickle

import numpy as np
import tables


def load_pickle(filename):
    with open(filename, 'rb') as in_file:
        inv_idx = pickle.load(in_file)

    return inv_idx


def save_pickle(inv_idx, filename):
    with open(filename, 'wb') as out_file:
        pickle.dump(inv_idx, out_file)


def numpy_to_hdf(obj, filename):
    obj_name = os.path.splitext(os.path.basename(filename))[0]
    with tables.open_file(filename, 'w') as f:
        atom = tables.Atom.from_dtype(obj.dtype)
        arr = f.create_carray(f.root, obj_name, atom, obj.shape)
        arr[:] = obj


def hdf_to_numpy(filename, compression_level=0, compression_lib='zlib'):
    obj_name = os.path.splitext(os.path.basename(filename))[0]
    filters = tables.Filters(complevel=compression_level, complib=compression_lib)
    with tables.open_file(filename, 'r', filters=filters) as f:
        arr = np.array(getattr(f.root, obj_name).read())

    return arr