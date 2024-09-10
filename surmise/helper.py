import numpy as np
import dill


def cast_f64_dtype(x):
    return np.array(x, dtype=np.float64)


def save_file(obj, filename):
    with open(filename, 'wb') as f:
        dill.dump(obj, f)


def load_file(filename, obj_type):
    with open(filename, 'rb') as f:
        loaded_file = dill.load(f)

    assert hasattr(loaded_file, '__module__')
    assert hasattr(loaded_file, '__class__')
    if '.'.join((loaded_file.__module__,
                 loaded_file.__class__.__name__)) != obj_type:
        raise TypeError('The file loaded should be of class {:s}.'.format(obj_type))
    return loaded_file
