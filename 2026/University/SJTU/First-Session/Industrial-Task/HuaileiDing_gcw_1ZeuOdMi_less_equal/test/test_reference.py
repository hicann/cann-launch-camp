#!/usr/bin/env python3
"""Host-side tests for the broadcasting/indexing contract used by the kernel."""

import unittest

import numpy as np

from test.generate_cases import build_cases


MAX_DIMS = 16
UINT32_MAX = (1 << 32) - 1


def make_tiling(shape1, shape2):
    rank = max(len(shape1), len(shape2))
    if rank > MAX_DIMS:
        raise ValueError("rank exceeds kernel limit")
    padded1 = (1,) * (rank - len(shape1)) + tuple(shape1)
    padded2 = (1,) * (rank - len(shape2)) + tuple(shape2)
    out = []
    for dim1, dim2 in zip(padded1, padded2):
        if dim1 < 0 or dim2 < 0 or (dim1 != dim2 and dim1 != 1 and dim2 != 1):
            raise ValueError("incompatible shapes")
        out.append(dim2 if dim1 == 1 else dim1)

    def strides(shape):
        result = [0] * rank
        stride = 1
        for axis in range(rank - 1, -1, -1):
            dim = shape[axis]
            result[axis] = 0 if dim == 1 and out[axis] != 1 else stride
            stride *= dim
        return result

    length = int(np.prod(out, dtype=np.uint64)) if out else 1
    if length > UINT32_MAX:
        raise ValueError("output exceeds uint32 indexing")
    return tuple(out), strides(padded1), strides(padded2), length


def kernel_reference(x1, x2):
    out_shape, stride1, stride2, length = make_tiling(x1.shape, x2.shape)
    flat1 = x1.reshape(-1)
    flat2 = x2.reshape(-1)
    result = np.empty(length, dtype=bool)
    for output_index in range(length):
        remaining = output_index
        index1 = index2 = 0
        for axis in range(len(out_shape) - 1, -1, -1):
            coordinate = remaining % out_shape[axis]
            remaining //= out_shape[axis]
            index1 += coordinate * stride1[axis]
            index2 += coordinate * stride2[axis]
        result[output_index] = flat1[index1] <= flat2[index2]
    return result.reshape(out_shape)


class LessEqualReferenceTest(unittest.TestCase):
    def test_all_generated_non_empty_cases(self):
        for name, x1, x2 in build_cases():
            if np.broadcast_shapes(x1.shape, x2.shape) == (0, 3):
                continue
            with self.subTest(name=name):
                np.testing.assert_array_equal(kernel_reference(x1, x2), np.less_equal(x1, x2))

    def test_empty_broadcast(self):
        out, stride1, stride2, length = make_tiling((0, 3), (1, 3))
        self.assertEqual(out, (0, 3))
        self.assertEqual(length, 0)
        self.assertEqual(stride1, [3, 1])
        self.assertEqual(stride2, [0, 1])

    def test_scalar(self):
        out, stride1, stride2, length = make_tiling((), ())
        self.assertEqual((out, stride1, stride2, length), ((), [], [], 1))

    def test_rejects_incompatible_shapes(self):
        with self.assertRaises(ValueError):
            make_tiling((2, 3), (4, 3))

    def test_rejects_rank_over_limit(self):
        with self.assertRaises(ValueError):
            make_tiling((1,) * 17, (1,))


if __name__ == "__main__":
    unittest.main()
