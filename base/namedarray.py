from typing import Dict, Tuple, List, Callable
import ast
import copy
from collections.abc import Mapping
from copy import deepcopy
from typing import Dict, Optional, Tuple, List, Callable
import ast
import enum
import itertools
import logging
import numpy as np
import pickle
import torch
import types
import warnings

import base.numpy_utils


class NamedArrayLoadingError(Exception):
    pass


class NamedArrayEncodingMethod(bytes, enum.Enum):
    """Encoding protocol of a NamedArray object.

    Principle: compress huge non-random data and pickle small data.

    TL;DR:
    + InferenceStream
        ++ If your observation include images, use PICKLE_COMPRESS.
        ++ Otherwise, use PICKLE or PICKLE_DICT.
    + SampleStream:
        ++ If your observation include images, use OBS_COMPRESS or COMPRESS_EXPECT_POLICY_STATE.
        ++ Otherwise,
            +++ if the amount of data is huge (say, multi-agent envs), use COMPRESS_EXPECT_POLICY_STATE.
            +++ otherwise, use PICKLE or PICKLE_DICT.
    """
    PICKLE_DICT = b"0001"  # Convert NamedArray to dict, then pickle.
    PICKLE = b"0002"  # Directly pickle.
    RAW_BYTES = b"0003"  # Send raw bytes of flattened numpy arrays.
    RAW_COMPRESS = b"0004"  # Send compressed bytes of flattened numpy arrays.
    COMPRESS_PICKLE = b"0005"  # Convert numpy array to compressed bytes, then pickle.
    PICKLE_COMPRESS = b"0006"  # Pickle, then compress pickled bytes.
    OBS_COMPRESS = b"0007"  # Convert flattened numpy array to bytes and only compress observation.
    COMPRESS_EXCEPT_POLICY_STATE = b"0008"  # Compress all bytes except for policy states, which are basically random numbers.


logger = logging.getLogger("NamedArray")


def _namedarray_op(op):

    def fn(self, value):
        if not (isinstance(value, NamedArray) and  # Check for matching structure.
                getattr(value, "_fields", None) == self._fields):
            if not isinstance(value, NamedArray):
                # Repeat value for each but respect any None.
                value = tuple(None if s is None else value for s in self)
            else:
                raise ValueError('namedarray - set an item with a different data structure')
        try:
            xs = {}
            for j, ((k, s), v) in enumerate(zip(self.items(), value)):
                if s is not None and v is not None:
                    exec(f"xs[k] = (s {op} v)")
                else:
                    exec(f"xs[k] = None")
        except (ValueError, IndexError, TypeError) as e:
            print(s.shape, v.shape)
            raise Exception(f"{type(e).__name__} occured in {self.__class__.__name__}"
                            " at field "
                            f"'{self._fields[j]}': {e}") from e
        return NamedArray(**xs)

    return fn


def _namedarray_iop(iop):

    def fn(self, value):
        if not (isinstance(value, NamedArray) and  # Check for matching structure.
                getattr(value, "_fields", None) == self._fields):
            if not isinstance(value, NamedArray):
                # Repeat value for each but respect any None.
                value = {k: None if s is None else value for k, s in self.items()}
            else:
                raise ValueError('namedarray - set an item with a different data structure')
        try:
            for j, (k, v) in enumerate(zip(self.keys(), value.values())):
                if self[k] is not None and v is not None:
                    exec(f"self[k] {iop} v")
        except (ValueError, IndexError, TypeError) as e:
            raise Exception(f"{type(e).__name__} occured in {self.__class__.__name__}"
                            " at field "
                            f"'{self._fields[j]}': {e}") from e
        return self

    return fn


def dumps(namedarray_obj, method="pickle_dict"):
    if 'compress' in method:
        # try:
        import blosc
        # except ModuleNotFoundError:

        #     class blosc:

        #         def compress(x, *args, **kwargs):
        #             return x

        #     warnings.warn("Module `blosc` not found in the image. Abort NamedArray compression.")

    def _namedarray_to_bytes_list(x, compress: bool, compress_condition: Callable[[str], bool]):
        flattened_entries = flatten(x)
        flattened_bytes = []
        for k, v in flattened_entries:
            k_ = k.encode('ascii')
            dtype_ = v_ = shape_ = b''
            if v is not None:
                dtype_ = base.numpy_utils.encode_dtype(v.dtype).encode('ascii')
                v_ = v.tobytes()
                shape_ = str(tuple(v.shape)).encode('ascii')
                if compress and compress_condition(k):
                    v_ = blosc.compress(v_, typesize=4, cname='lz4')
            flattened_bytes.append((k_, dtype_, shape_, v_))
        return list(itertools.chain.from_iterable(flattened_bytes))

    if method == "pickle_dict":
        bytes_list = [
            NamedArrayEncodingMethod.PICKLE_DICT.value,
            pickle.dumps((namedarray_obj.__class__.__name__, namedarray_obj.to_dict())),
        ]
    elif method == "pickle":
        bytes_list = [NamedArrayEncodingMethod.PICKLE.value, pickle.dumps(namedarray_obj)]
    elif method == 'raw_bytes':
        bytes_list = [NamedArrayEncodingMethod.RAW_BYTES.value] + _namedarray_to_bytes_list(
            namedarray_obj, False, lambda x: False)
    elif method == 'raw_compress':
        bytes_list = [NamedArrayEncodingMethod.RAW_COMPRESS.value] + _namedarray_to_bytes_list(
            namedarray_obj, True, lambda x: True)
    elif method == 'compress_pickle':
        bytes_list = [
            NamedArrayEncodingMethod.COMPRESS_PICKLE.value,
            pickle.dumps(_namedarray_to_bytes_list(namedarray_obj, True, lambda x: True))
        ]
    elif method == 'pickle_compress':
        bytes_list = [
            NamedArrayEncodingMethod.PICKLE_COMPRESS.value,
            blosc.compress(pickle.dumps(namedarray_obj), typesize=4, cname='lz4')
        ]
    elif method == 'obs_compress':
        bytes_list = [NamedArrayEncodingMethod.OBS_COMPRESS.value] + _namedarray_to_bytes_list(
            namedarray_obj, True, lambda x: ('obs' in x))
    elif method == 'compress_except_policy_state':
        bytes_list = [NamedArrayEncodingMethod.COMPRESS_EXCEPT_POLICY_STATE.value
                      ] + _namedarray_to_bytes_list(namedarray_obj, True, lambda x: ('policy_state' not in x))
    else:
        raise NotImplementedError(
            f"Unknown method {method}. Available are {[m.name.lower() for m in NamedArrayEncodingMethod]}.")

    return bytes_list + [pickle.dumps(dict(**namedarray_obj.metadata))]


def loads(b):
    # safe import
    if b[0] in [b'0004', b'0004', b'0005', b'0006', b'0007', b'0008']:
        # try:
        import blosc
        # except ModuleNotFoundError:

        #     class blosc:

        #         def decompress(x, *args, **kwargs):
        #             return x

    def _parse_namedarray_from_bytes_list(xs, compressed: bool, compress_condition: Callable[[str], int]):
        flattened = []
        for i in range(len(xs) // 4):
            k = xs[4 * i].decode('ascii')
            if xs[4 * i + 1] != b'':
                buf = xs[4 * i + 3]
                if compressed and compress_condition(k):
                    buf = blosc.decompress(buf)
                v = np.frombuffer(buf, dtype=base.numpy_utils.decode_dtype(
                    xs[4 * i + 1].decode('ascii'))).reshape(*ast.literal_eval(xs[4 * i + 2].decode('ascii')))
            else:
                v = None
            flattened.append((k, v))
        return from_flattened(flattened)

    if b[0] == NamedArrayEncodingMethod.PICKLE_DICT.value:
        class_name, values = pickle.loads(b[1])
        namedarray_obj = from_dict(values=values)
    elif b[0] == NamedArrayEncodingMethod.PICKLE.value:
        namedarray_obj = pickle.loads(b[1])
    elif b[0] == NamedArrayEncodingMethod.RAW_BYTES.value:
        namedarray_obj = _parse_namedarray_from_bytes_list(b[1:-1], False, lambda x: False)
    elif b[0] == NamedArrayEncodingMethod.RAW_COMPRESS.value:
        namedarray_obj = _parse_namedarray_from_bytes_list(b[1:-1], True, lambda x: True)
    elif b[0] == NamedArrayEncodingMethod.COMPRESS_PICKLE.value:
        namedarray_obj = _parse_namedarray_from_bytes_list(pickle.loads(b[1]), True, lambda x: True)
    elif b[0] == NamedArrayEncodingMethod.PICKLE_COMPRESS.value:
        namedarray_obj = pickle.loads(blosc.decompress(b[1]))
    elif b[0] == NamedArrayEncodingMethod.OBS_COMPRESS.value:
        namedarray_obj = _parse_namedarray_from_bytes_list(b[1:-1], True, lambda x: ('obs' in x))
    elif b[0] == NamedArrayEncodingMethod.COMPRESS_EXCEPT_POLICY_STATE.value:
        namedarray_obj = _parse_namedarray_from_bytes_list(b[1:-1], True, lambda x: ('policy_state' not in x))
    else:
        raise NotImplementedError(f"Unknown NamedArrayEncodingMethod value {b[:4]}. "
                                  f"Existing are {[m for m in NamedArrayEncodingMethod]}.")

    namedarray_obj.clear_metadata()
    metadata = pickle.loads(b[-1])
    namedarray_obj.register_metadata(**metadata)

    return namedarray_obj


class NamedArray:
    """A class decorator modified from the `namedarraytuple` class in rlpyt repo,
    referring to
    https://github.com/astooke/rlpyt/blob/master/rlpyt/utils/collections.py#L16.

    NamedArray supports dict-like unpacking and string indexing, and exposes integer slicing reads
    and writes applied to all contained objects, which must share
    indexing (__getitem__) behavior (e.g. numpy arrays or torch tensors).

    Note that namedarray supports nested structure.,
    i.e., the elements of a NamedArray could also be NamedArray.

    Example:
    >>> class Point(NamedArray):
    ...     def __init__(self,
    ...         x: np.ndarray,
    ...         y: np.ndarray,
    ...         ):
    ...         super().__init__(x=x, y=y)
    >>> p=Point(np.array([1,2]), np.array([3,4]))
    >>> p
    Point(x=array([1, 2]), y=array([3, 4]))
    >>> p[:-1]
    Point(x=array([1]), y=array([3]))
    >>> p[0]
    Point(x=1, y=3)
    >>> p.x
    array([1, 2])
    >>> p['y']
    array([3, 4])
    >>> p[0] = 0
    >>> p
    Point(x=array([0, 2]), y=array([0, 4]))
    >>> p[0] = Point(5, 5)
    >>> p
    Point(x=array([5, 2]), y=array([5, 4]))
    >>> 'x' in p
    True
    >>> list(p.keys())
    ['x', 'y']
    >>> list(p.values())
    [array([5, 2]), array([5, 4])]
    >>> for k, v in p.items():
    ...     print(k, v)
    ...
    x [5 2]
    y [5 4]
    >>> def foo(x, y):
    ...     print(x, y)
    ...
    >>> foo(**p)
    [5 2] [5 4]
    """
    _reserved_slots = ["_NamedArray__metadata", "_fields"]

    def __init__(self, **kwargs):
        """

        Args:
            data: key-value following {field_name: otherNamedArray/None/np.ndarray/torch.Tensor}
        """
        self._fields = tuple(sorted(kwargs.keys()))
        self.__metadata = types.MappingProxyType({})
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def metadata(self):
        return self.__metadata

    def register_metadata(self, **kwargs):
        for k in self._fields:
            if k in kwargs.keys():
                raise KeyError("Keys of metadata should be different from data fields!")

        self.__metadata = types.MappingProxyType({**self.__metadata, **kwargs})

    def pop_metadata(self, key):
        metadatadict = dict(self.__metadata)
        value = metadatadict.pop(key)
        self.__metadata = types.MappingProxyType(metadatadict)
        return value

    def clear_metadata(self):
        self.__metadata = types.MappingProxyType({})

    def __iter__(self):
        for k in self._fields:
            yield getattr(self, k)

    def __setattr__(self, loc, value):
        """Set attributes in a `NamedArray` object.
        
        Unknown fields cannot be created.
            
        Args:
            loc (str): attribute name to be set.
        """
        # if not (loc in NamedArray._reserved_slots or loc in self._fields):
        #     raise AttributeError(f"Cannot add new attribute '{loc}' to instance of {type(self).__name__}")
        super().__setattr__(loc, value)

    def __getitem__(self, loc):
        """If the index is string, return getattr(self, index).
        If the index is integer/slice, return a new dataclass instance containing
        the selected index or slice from each field.

        Args:
            loc (str or slice): Key or indices to get.

        Raises:
            Exception: To locate in which field the error occurs.

        Returns:
            Any: An element of the dataclass or a new dataclass
                object composed of the subarrays.
        """
        if isinstance(loc, str):
            # str indexing like in dict
            return getattr(self, loc)
        else:
            sliced_namedarray = dict()
            try:
                for s in self._fields:
                    if self[s] is None:
                        sliced_namedarray[s] = None
                    else:
                        sliced_namedarray[s] = self[s][loc]
            except IndexError as e:
                raise Exception(f"IndexError occured when slicing `NamedArray`."
                                f"Field {s} with shape {self[s].shape} and slice {loc}.") from e
            return self.__class__(**{s: None if self[s] is None else self[s][loc] for s in self._fields})

    def __setitem__(self, loc, value):
        """If input value is the same dataclass type, iterate through its
        fields and assign values into selected index or slice of corresponding
        field.  Else, assign whole of value to selected index or slice of
        all fields. Ignore fields that are both None.

        Args:
            loc (str or slice): Key or indices to set.
            value (Any): A dataclass instance with the same structure
                or elements of the dataclass object.

        Raises:
            Exception: To locate in which field the error occurs.
        """
        if isinstance(loc, str):
            setattr(self, loc, value)
        else:
            if not (isinstance(value, NamedArray) and  # Check for matching structure.
                    getattr(value, "_fields", None) == self._fields):
                if not isinstance(value, NamedArray):
                    # Repeat value for each but respect any None.
                    value = tuple(None if s is None else value for s in self)
                else:
                    raise ValueError('namedarray - set an item with a different data structure')
            try:
                for j, (s, v) in enumerate(zip(self, value)):
                    if s is not None and v is not None:
                        s[loc] = v
            except (ValueError, IndexError, TypeError) as e:
                raise Exception(f"Error occured occured in {self.__class__.__name__} when assigning value"
                                " at field "
                                f"'{self._fields[j]}': {e}") from e

    def __contains__(self, key):
        """Checks presence of field name (unlike tuple; like dict).

        Args:
            key (str): The queried field name.

        Returns:
            bool: Query result.
        """
        return key in self._fields

    def __getstate__(self):
        return {'__metadata': dict(**self.metadata), **{k: v for k, v in self.items()}}

    def __setstate__(self, state):
        self.__init__(**{k: v for k, v in state.items() if k != '__metadata'})
        if state['__metadata'] is not None:
            self.clear_metadata()
            self.register_metadata(**state['__metadata'])

    def values(self):
        for v in self:
            yield v

    def keys(self):
        for k in self._fields:
            yield k

    def __len__(self):
        return len(self._fields)

    def length(self, dim=0):
        for k, v in self.items():
            if v is None:
                continue
            elif isinstance(v, (np.ndarray, torch.Tensor)):
                if dim < v.ndim:
                    return v.shape[dim]
                else:
                    continue
            else:
                # TODO: Do we care about nested?
                continue
        else:
            raise IndexError(f"No entry has shape on dim={dim}.")

    def unique_of(self, field, exclude_values=(None,)):
        """Get the unique value of a field
        """
        unique_values = np.unique(self[field])
        unique_values = unique_values[np.in1d(unique_values, exclude_values, invert=True)]
        if len(unique_values) != 1:
            return None
        else:
            return unique_values[0]

    def average_of(self, field, ignore_negative=True):
        """Get the average value of the sample
        Returns:
            version: average version of the sample in trainer steps. None if no version is specified for any data.
        """
        values = self[field]
        if len(values) > 0:
            if ignore_negative:
                return np.nanmean(np.where(values >= 0, values, np.nan))
            else:
                return values.mean()
        else:
            return None

    def max_of(self, field, ignore_negative=True):
        """Get the average value of the sample
        Returns:
            version: average version of the sample in trainer steps. None if no version is specified for any data.
        """
        values = self[field]
        if len(values) > 0:
            if ignore_negative:
                return np.nanmax(np.where(values >= 0, values, np.nan))
            else:
                return values.max()
        else:
            return None

    def min_of(self, field, ignore_negative=True):
        """Get the average value of the sample
        Returns:
            version: average version of the sample in trainer steps. None if no version is specified for any data.
        """
        values = self[field]
        if len(values) > 0:
            if ignore_negative:
                return np.nanmin(np.where(values >= 0, values, np.nan))
            else:
                return values.min()
        else:
            return None

    def items(self):
        """Iterate over ordered (field_name, value) pairs.

        Yields:
            tuple[str,Any]: (field_name, value) pairs
        """
        for k, v in zip(self._fields, self):
            yield k, v

    def to_dict(self):
        result = {}
        for k, v in self.items():
            if isinstance(v, NamedArray):
                result[k] = v.to_dict()
            elif v is None:
                result[k] = None
            else:
                result[k] = v
        return result

    @property
    def shape(self):
        return recursive_apply(self, lambda x: x.shape).to_dict()

    def size(self):
        return self.shape

    def __str__(self):
        return f"{self.__class__.__name__}({', '.join(k+'='+repr(v) for k, v in self.items())})"

    def __deepcopy__(self, memo={}):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        metadata = {}
        for k, v in self.__dict__.items():
            if isinstance(v, types.MappingProxyType):
                metadata = copy.deepcopy(dict(v), memo)
                continue
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        result.clear_metadata()
        result.register_metadata(**metadata)
        return result

    __add__ = _namedarray_op('+')
    __sub__ = _namedarray_op('-')
    __mul__ = _namedarray_op('*')
    __truediv__ = _namedarray_op('/')
    __iadd__ = _namedarray_iop('+=')
    __isub__ = _namedarray_iop('-=')
    __imul__ = _namedarray_iop('*=')
    __itruediv__ = _namedarray_iop('/=')
    __repr__ = __str__


def from_dict(values: Dict):
    """Create namedarray object from Nested Dict of arrays.
    Args:
        values: Nested key-value object of data. value should of type None, Numpy Array, or Torch Tensor.
                Return None if length of value is 0.
    Returns:
        NamedArray with the same data structure as input. If values is None, return None.
    Example:
    >>> a = from_dict({"x": np.array([1, 2]), "y": np.array([3,4])})
    >>> a.x
    array([1, 2])
    >>> a.y
    array([3, 4])
    >>> a[1:]
    NamedArray(x=[2],y=[4])
    >>> obs = {"state":{"speed": np.array([1, 2, 3]), "position": np.array([4, 5])}, "vision": np.array([[7],[8],[9]])}
    >>> obs_na = from_dict(obs)
    >>> obs_na
    NamedArray(state=NamedArray(position=[4 5],speed=[1 2 3]),vision=[[7]
     [8]
     [9]])
    >>> obs_na.state
    NamedArray(position=[4 5],speed=[1 2 3])
    """
    if values is None or len(values) == 0:
        return None
    for k, v in values.items():
        if isinstance(v, dict):
            values[k] = from_dict(v)
    return NamedArray(**values)


def array_like(x, value=0):
    if isinstance(x, NamedArray):
        return NamedArray(**{k: array_like(v, value) if v is not None else None for k, v in x.items()})
    else:
        if isinstance(x, np.ndarray):
            data = np.zeros_like(x)
        else:
            assert isinstance(x, torch.Tensor), ('Currently, namedarray only supports'
                                                 f' torch.Tensor and numpy.array (input is {type(x)})')
            data = torch.zeros_like(x)
        if value != 0:
            data[:] = value
        return data


def __array_filter_none(xs):
    is_not_nones = [x is not None for x in xs]
    if all(is_not_nones) or all(x is None for x in xs):
        return
    else:
        example_x = xs[is_not_nones.index(True)]
        for i, x in enumerate(xs):
            xs[i] = array_like(example_x) if x is None else x


def recursive_aggregate(xs, aggregate_fn):
    """Recursively aggregate a list of namedarray instances.
    Typically recursively stacking or concatenating.

    Args:
        xs (List[Any]): A list of namedarrays or
            appropriate aggregation targets (e.g. numpy.ndarray).
        aggregate_fn (function): The aggregation function to be applied.

    Returns:
        Any: The aggregated result with the same data type of elements in xs.
    """
    __array_filter_none(xs)
    if isinstance(xs[0], NamedArray):
        entries = dict()
        for k in xs[0].keys():
            try:
                entries[k] = recursive_aggregate([x[k] for x in xs], aggregate_fn)
            except Exception as e:
                err_msg = f"`recursive_aggregate` fails at an entry named `{k}`."
                if not all([type(x[k]) == type(xs[0][k]) for x in xs]):
                    err_msg += f" Types of elements are not the same: {[type(x[k]) for x in xs]}."
                else:
                    if isinstance(xs[0][k], NamedArray):
                        err_msg += " Elements are all `NamedArray`s. Backtrace to the above level."
                if not any([isinstance(x[k], NamedArray) for x in xs]):
                    specs = []
                    for x in xs:
                        specs.append(f"({x[k].dtype}, {tuple(x[k].shape)})")
                    # err_msg += f" Specs of elements to be aggregated are: [{', '.join(specs)}]."
                raise RuntimeError(err_msg) from e
        return NamedArray(**entries)
    elif xs[0] is None:
        return None
    else:
        return aggregate_fn(xs)


def recursive_apply(x, fn):
    """Recursively apply a function to a namedarray x.

    Args:
        x (Any): The instance of a namedarray subclass
            or an appropriate target to apply fn.
        fn (function): The function to be applied.
    """
    if isinstance(x, NamedArray):
        entries = dict()
        for k, v in x.items():
            try:
                entries[k] = recursive_apply(v, fn)
            except Exception as e:
                err_msg = f"`recursive_apply` fails at an entry named `{k}`"
                if isinstance(v, NamedArray):
                    err_msg += ", which is a `NamedArray`. Backtrace to the above level."
                else:
                    err_msg += f" ({v.dtype}, {tuple(v.shape)})."
                raise RuntimeError(err_msg) from e
        return NamedArray(**entries)
    elif x is None:
        return None
    else:
        return fn(x)


def flatten(x: NamedArray) -> List[Tuple]:
    """Flatten a NamedArray object to a list containing structured names and values."""
    flattened_entries = []
    for k, v in x.items():
        if isinstance(v, NamedArray):
            flattened_entries += [(f"{k}." + k_, v_) for k_, v_ in flatten(v)]
        else:
            flattened_entries.append((k, v))
    return flattened_entries


def from_flattened(flattened_entries):
    """Construct a NamedArray from flattened names and values."""
    keys, values = zip(*flattened_entries)
    entries = dict()
    idx = 0
    while idx < len(keys):
        k, v = keys[idx], values[idx]
        if '.' not in k:
            entries[k] = v
            idx += 1
        else:
            prefix = k.split('.')[0]
            span_end = idx + 1
            while span_end < len(keys) and keys[span_end].startswith(f"{prefix}."):
                span_end += 1
            subentries = [(keys[j][len(prefix) + 1:], values[j]) for j in range(idx, span_end)]
            entries[prefix] = from_flattened(subentries)
            idx = span_end
    return from_dict(entries)


def size_bytes(x):
    """Return the size of a namedarray in bytes."""
    return sum([
        base.numpy_utils.dtype_to_num_bytes(v.dtype) * np.prod(v.shape) if v is not None else 0
        for _, v in flatten(x)
    ])
