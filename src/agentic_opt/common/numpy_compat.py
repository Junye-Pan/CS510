
from __future__ import annotations

import builtins
import math
from copy import deepcopy
from itertools import zip_longest


def _is_sequence(value):
    return isinstance(value, (list, tuple))


def _raw(value):
    return value._data if isinstance(value, ndarray) else value


def _coerce(value):
    value = _raw(value)
    if _is_sequence(value):
        return [_coerce(item) for item in value]
    if isinstance(value, bool):
        return bool(value)
    return float(value)


def _shape_of(data):
    if isinstance(data, list):
        if not data:
            return (0,)
        return (len(data),) + _shape_of(data[0])
    return ()


def _wrap(value):
    if isinstance(value, list):
        return ndarray(value)
    return value


def _flatten(data):
    if isinstance(data, list):
        for item in data:
            yield from _flatten(item)
    else:
        yield data


def _slice_data(data, keys):
    if not isinstance(keys, tuple):
        keys = (keys,)

    def rec(current, remaining):
        if not remaining:
            return current
        head, *tail = remaining
        if head is None:
            return [rec(current, tuple(tail))]
        if isinstance(head, slice):
            if not isinstance(current, list):
                raise TypeError("cannot slice a scalar")
            return [rec(item, tuple(tail)) for item in current[head]]
        return rec(current[head], tuple(tail))

    return rec(data, keys)


def _assign_nested(target, keys, value):
    if not isinstance(keys, tuple):
        keys = (keys,)
    if len(keys) == 1:
        key = keys[0]
        raw_value = _raw(value)
        target[key] = raw_value
        return
    head = keys[0]
    tail = keys[1:]
    if isinstance(head, slice):
        subset = target[head]
        raw_value = _raw(value)
        if isinstance(raw_value, list) and len(raw_value) == len(subset):
            for item, subvalue in zip(subset, raw_value):
                _assign_nested(item, tail, subvalue)
        else:
            for item in subset:
                _assign_nested(item, tail, raw_value)
        return
    _assign_nested(target[head], tail, value)


def _broadcast_shape(shape_a, shape_b):
    result = []
    for dim_a, dim_b in zip_longest(reversed(shape_a), reversed(shape_b), fillvalue=1):
        if dim_a == 1:
            result.append(dim_b)
        elif dim_b == 1:
            result.append(dim_a)
        elif dim_a == dim_b:
            result.append(dim_a)
        else:
            raise ValueError(f"incompatible shapes: {shape_a} and {shape_b}")
    return tuple(reversed(result))


def _broadcast_to(data, source_shape, target_shape):
    if source_shape == target_shape:
        return deepcopy(data)
    padded_shape = (1,) * (len(target_shape) - len(source_shape)) + source_shape
    padded_data = deepcopy(data)
    for _ in range(len(target_shape) - len(source_shape)):
        padded_data = [padded_data]

    def expand(node, src_shape, dst_shape):
        if not dst_shape:
            return deepcopy(node)
        src_dim = src_shape[0] if src_shape else 1
        dst_dim = dst_shape[0]
        next_src = src_shape[1:] if src_shape else ()
        next_dst = dst_shape[1:]
        if src_dim == dst_dim:
            return [expand(child, next_src, next_dst) for child in node]
        if src_dim == 1:
            child = node[0] if isinstance(node, list) else node
            expanded = expand(child, next_src, next_dst)
            return [deepcopy(expanded) for _ in range(dst_dim)]
        raise ValueError(f"cannot broadcast {source_shape} to {target_shape}")

    return expand(padded_data, padded_shape, target_shape)


def _elementwise_binary(left, right, op):
    left = _raw(left)
    right = _raw(right)
    left_shape = _shape_of(left)
    right_shape = _shape_of(right)
    target_shape = _broadcast_shape(left_shape, right_shape)
    left_data = _broadcast_to(left, left_shape, target_shape)
    right_data = _broadcast_to(right, right_shape, target_shape)

    def rec(a, b):
        if isinstance(a, list):
            return [rec(x, y) for x, y in zip(a, b)]
        return op(a, b)

    return _wrap(rec(left_data, right_data))


def _elementwise_unary(value, op):
    value = _raw(value)
    if isinstance(value, list):
        return _wrap([_raw(_elementwise_unary(item, op)) for item in value])
    return op(value)


def _reduce_axis(data, axis, reducer):
    if axis == 0:
        if not isinstance(data, list):
            return data
        if not data:
            return []
        if not isinstance(data[0], list):
            return reducer(data)
        width = len(data[0])
        return [_reduce_axis([row[index] for row in data], 0, reducer) for index in range(width)]
    return [_reduce_axis(item, axis - 1, reducer) for item in data]


def _sum_values(values):
    total = 0.0
    for value in values:
        total += float(value)
    return total


class ndarray:
    def __init__(self, data):
        self._data = _coerce(data)
        self.shape = _shape_of(self._data)

    def copy(self):
        return ndarray(self._data)

    def tolist(self):
        return deepcopy(self._data)

    def astype(self, dtype):
        del dtype
        return ndarray(self._data)

    @property
    def T(self):
        if len(self.shape) != 2:
            return ndarray(self._data)
        rows, cols = self.shape
        return ndarray([[self._data[row][col] for row in range(rows)] for col in range(cols)])

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        return _wrap(_slice_data(self._data, key))

    def __setitem__(self, key, value):
        _assign_nested(self._data, key, value)

    def __add__(self, other):
        return _elementwise_binary(self, other, lambda a, b: a + b)

    def __radd__(self, other):
        return _elementwise_binary(other, self, lambda a, b: a + b)

    def __sub__(self, other):
        return _elementwise_binary(self, other, lambda a, b: a - b)

    def __rsub__(self, other):
        return _elementwise_binary(other, self, lambda a, b: a - b)

    def __mul__(self, other):
        return _elementwise_binary(self, other, lambda a, b: a * b)

    def __rmul__(self, other):
        return _elementwise_binary(other, self, lambda a, b: a * b)

    def __truediv__(self, other):
        return _elementwise_binary(self, other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        return _elementwise_binary(other, self, lambda a, b: a / b)

    def __pow__(self, other):
        return _elementwise_binary(self, other, lambda a, b: a**b)

    def __neg__(self):
        return _elementwise_unary(self, lambda value: -value)

    def __lt__(self, other):
        return _elementwise_binary(self, other, lambda a, b: a < b)

    def __le__(self, other):
        return _elementwise_binary(self, other, lambda a, b: a <= b)

    def __gt__(self, other):
        return _elementwise_binary(self, other, lambda a, b: a > b)

    def __ge__(self, other):
        return _elementwise_binary(self, other, lambda a, b: a >= b)

    def sum(self, axis=None):
        return sum(self, axis=axis)


def array(data, dtype=None):
    del dtype
    return ndarray(data)


def asarray(data, dtype=None):
    del dtype
    return ndarray(data)


def sum(values, axis=None):
    values = asarray(values)
    if axis is None:
        return float(_sum_values(_flatten(values._data)))
    reduced = _reduce_axis(values._data, axis, _sum_values)
    return _wrap(reduced)


def min(values, axis=None):
    values = asarray(values)
    if axis is None:
        return builtins.min(_flatten(values._data))
    reduced = _reduce_axis(values._data, axis, builtins.min)
    return _wrap(reduced)


def max(values, axis=None):
    values = asarray(values)
    if axis is None:
        return builtins.max(_flatten(values._data))
    reduced = _reduce_axis(values._data, axis, builtins.max)
    return _wrap(reduced)


def any(values, axis=None):
    values = asarray(values)
    if axis is None:
        return builtins.any(_flatten(values._data))
    reduced = _reduce_axis(values._data, axis, builtins.any)
    return _wrap(reduced)


def all(values, axis=None):
    values = asarray(values)
    if axis is None:
        return builtins.all(_flatten(values._data))
    reduced = _reduce_axis(values._data, axis, builtins.all)
    return _wrap(reduced)


def sqrt(values):
    return _elementwise_unary(values, math.sqrt)


def abs(values):
    return _elementwise_unary(values, builtins.abs)


def maximum(left, right):
    return _elementwise_binary(left, right, lambda a, b: a if a >= b else b)


def isfinite(values):
    return _elementwise_unary(values, math.isfinite)


def isclose(left, right, rtol=1e-05, atol=1e-08):
    return _elementwise_binary(left, right, lambda a, b: builtins.abs(a - b) <= atol + rtol * builtins.abs(b))


def concatenate(values, axis=0):
    arrays = [asarray(value)._data for value in values]
    if axis == 0:
        result = []
        for value in arrays:
            result.extend(deepcopy(value))
        return ndarray(result)
    if axis == 1:
        result = []
        for rows in zip(*arrays):
            combined = []
            for row in rows:
                combined.extend(deepcopy(row))
            result.append(combined)
        return ndarray(result)
    raise NotImplementedError("concatenate only supports axis 0 or 1")


def clip(values, lower, upper):
    return _elementwise_unary(values, lambda value: builtins.max(lower, builtins.min(upper, value)))


def zeros(shape, dtype=float):
    del dtype
    if isinstance(shape, int):
        shape = (shape,)
    if not shape:
        return 0.0
    return ndarray([zeros(shape[1:]) for _ in range(shape[0])])


class _Linalg:
    @staticmethod
    def norm(values, axis=None):
        values = asarray(values)
        squared = values * values
        return sqrt(sum(squared, axis=axis))


linalg = _Linalg()
float64 = float
inf = float("inf")
