from typing import Dict
import numpy as np


def split_to_shapes(x: np.ndarray, shapes: Dict, axis: int = -1):
    """Split an array and reshape to desired shapes.

    Args:
        x (np.ndarray): The array to be splitted
        shapes (Dict): Dict of shapes (tuples) specifying how to split.
        axis (int): Split dimension.

    Returns:
        List: Splitted observations.
    """
    axis = len(x.shape) + axis if axis < 0 else axis
    split_lengths = [np.prod(shape) for shape in shapes.values()]
    assert x.shape[axis] == sum(split_lengths)
    accum_split_lengths = [sum(split_lengths[:i]) for i in range(1, len(split_lengths))]
    splitted_x = np.split(x, accum_split_lengths, axis)
    return {
        k: x.reshape(*x.shape[:axis], *shape, *x.shape[axis + 1:])
        for x, (k, shape) in zip(splitted_x, shapes.items())
    }


def moving_average(x: np.ndarray, window_size: int):
    """Return the moving average of a 1D numpy array.
    """
    if len(x.shape) != 1:
        raise ValueError("Moving average works only on 1D arrays!")
    if window_size > x.shape[0]:
        raise ValueError("Can't average over a window size larger than array length!")
    return np.convolve(np.ones(window_size, dtype=np.float32) / window_size, x, 'valid')


def moving_maximum(x: np.ndarray, window_size: int):
    if len(x.shape) != 1:
        raise ValueError("Moving maximum works only on 1D arrays!")
    if window_size > x.shape[0]:
        raise ValueError("Can't average over a window size larger than array length!")
    shape = (x.shape[0] - window_size + 1, window_size)
    strides = x.strides + (x.strides[-1],)
    rolling = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return np.max(rolling, axis=1)


def dtype_to_num_bytes(dtype: np.dtype) -> int:
    if dtype in [np.bool8, np.int8, np.uint8]:
        return 1
    if dtype in [np.uint16, np.int16, np.float16]:
        return 2
    if dtype in [np.uint32, np.int32, np.float32]:
        return 4
    if dtype in [np.uint64, np.int64, np.float64]:
        return 8
    if dtype in [np.float128]:
        return 16
    elif str(dtype).startswith("<U"):
        return int(str(dtype)[2:]) * 4
    return 4


def encode_dtype(dtype: np.dtype) -> str:
    if dtype == np.uint8 or dtype == bool:
        return "uint8"
    elif dtype == np.float32:
        return "float32"
    elif dtype == np.float64:
        return "float64"
    elif dtype == np.int32:
        return "int32"
    elif dtype == np.int64:
        return "int64"
    elif str(dtype).startswith("<U"):  # string array, e.g. policy_name
        return str(dtype)
    else:
        raise NotImplementedError(f"Data type to string not implemented: {dtype}.")


def decode_dtype(dtype_str: str) -> np.dtype:
    return np.dtype(dtype_str)


def set_non_zero_to_one(array):
    # Return a new copy of the array
    new_array = np.copy(array)
    new_array[new_array != 0] = 1
    return new_array
