import numpy as np
import dill


def cast_f64_dtype(x):
    return np.array(x, dtype=np.float64)


def save_file(obj, filename):
    with open(filename, 'wb') as f:
        dill.dump(obj, f)


def load_file(filename):
    with open(filename, 'rb') as f:
        loaded_file = dill.load(f)
    return loaded_file
