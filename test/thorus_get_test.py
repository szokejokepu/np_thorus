import unittest

import numpy as np

from src.Thorus import Thorus


class GetTestCase(unittest.TestCase):

    def test_get_1d(self):
        data = np.arange(2)
        t = Thorus.make(data)

        self.assertTrue(np.array_equal(data[0], t[0], equal_nan=True))
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))
        self.assertTrue(np.array_equal(data[0], t[2], equal_nan=True))
        self.assertTrue(np.array_equal(data[1], t[3], equal_nan=True))
        self.assertTrue(np.array_equal(data[-1], t[-1], equal_nan=True))
        self.assertTrue(np.array_equal(data[-2], t[-2], equal_nan=True))

        self.assertTrue(np.array_equal(data[0:1], t[0:1], equal_nan=True))
        self.assertTrue(np.array_equal(data[0:2], t[0:2], equal_nan=True))
        self.assertTrue(np.array_equal(data[1:2], t[1:2], equal_nan=True))
        self.assertTrue(np.array_equal([*data[1:2], *data[0:1]], t[1:3], equal_nan=True))
        self.assertTrue(np.array_equal([*data[1:2], *data[0:1]], t[-1:1], equal_nan=True))
        self.assertTrue(np.array_equal(data[1:2], t[-1:0], equal_nan=True))
        self.assertTrue(np.array_equal(data[0:1], t[2:3], equal_nan=True))

        self.assertTrue(np.array_equal(data[0:], t[0:], equal_nan=True))
        self.assertTrue(np.array_equal(data[:1], t[:1], equal_nan=True))
        self.assertTrue(np.array_equal([*data[:], *data[0:1]], t[:3], equal_nan=True))

    def test_get_1d_transposed(self):
        data = np.arange(3).reshape((1, 3))
        t = Thorus.make(data)

        self.assertTrue(np.array_equal(data[0], t[0], equal_nan=True))
        self.assertTrue(np.array_equal(data[0], t[2], equal_nan=True))
        self.assertTrue(np.array_equal(data[0, 0], t[0, 3], equal_nan=True))
        self.assertTrue(np.array_equal(data[-1], t[-1], equal_nan=True))
        self.assertTrue(np.array_equal(data[0:1], t[0:1], equal_nan=True))
        self.assertTrue(np.array_equal(data[-1:], t[-1:], equal_nan=True))
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))
        self.assertTrue(np.array_equal(data[1:3], t[1:3], equal_nan=True))
        self.assertTrue(np.array_equal(data[2:3], t[2:3], equal_nan=True))

        self.assertTrue(np.array_equal([*data[0, -1:], *data[0, 0:1]], t[0, -1:1], equal_nan=True))
        self.assertTrue(np.array_equal(data[0, -1:], t[0, -1:], equal_nan=True))
        self.assertTrue(np.array_equal(data[0, -1:], t[0, -1:0], equal_nan=True))
        self.assertTrue(np.array_equal(data[0, 1:2], t[0, 1:2], equal_nan=True))
        self.assertTrue(np.array_equal(data[0, 4:8], t[0, 4:8], equal_nan=True))
        self.assertTrue(np.array_equal(data[0, -2:-1], t[0, -2:-1], equal_nan=True))
        self.assertTrue(np.array_equal(data[0, -5:-1], t[0, -5:-1], equal_nan=True))

    def test_get_2d(self):
        data = np.arange(25).reshape((5, 5))
        t = Thorus.make(data)

        self.assertTrue(np.array_equal(data[0, 0], t[0, 0], equal_nan=True))
        self.assertTrue(np.array_equal(data[0, 0], t[0, 5], equal_nan=True))
        self.assertTrue(np.array_equal(data[0, 0], t[5, 0], equal_nan=True))
        self.assertTrue(np.array_equal(data[0, 0], t[5, 5], equal_nan=True))
        self.assertTrue(np.array_equal(data[-1, -1], t[-1, -1], equal_nan=True))
        self.assertTrue(np.array_equal(data[-1, -1], t[4, 4], equal_nan=True))

        self.assertTrue(np.array_equal(data[:, :], t[:, :], equal_nan=True))
        self.assertTrue(np.array_equal(data[3:, 3:], t[3:, 3:], equal_nan=True))

        self.assertTrue(np.array_equal(np.concatenate((data[0, :], data[0, :2]), axis=0), t[0, :7], equal_nan=True))
        self.assertTrue(np.array_equal(np.concatenate((data[:, 0], data[:2, 0]), axis=0), t[:7, 0], equal_nan=True))

        self.assertTrue(
            np.array_equal(np.concatenate((data[0:1, :], data[0:1, :2]), axis=1), t[0:1, :7], equal_nan=True))
        self.assertTrue(
            np.array_equal(np.concatenate((data[:, 0:1], data[:2, 0:1]), axis=0), t[:7, 0:1], equal_nan=True))

        self.assertTrue(
            np.array_equal(np.concatenate((data[0:1, 3:], data[0:1, :2]), axis=1), t[0:1, 3:7], equal_nan=True))
        self.assertTrue(
            np.array_equal(np.concatenate((data[3:, 0:1], data[:2, 0:1]), axis=0), t[3:7, 0:1], equal_nan=True))

        self.assertTrue(np.array_equal(np.concatenate((data[:, 3:], data[:, :2]), axis=1), t[:, 3:7], equal_nan=True))
        self.assertTrue(np.array_equal(np.concatenate((data[3:, :], data[:2, :]), axis=0), t[3:7, :], equal_nan=True))

        self.assertTrue(np.array_equal(np.concatenate((np.concatenate((data[3:, 3:], data[:2, 3:]), axis=0),
                                                       np.concatenate((data[3:, :2], data[:2, :2]), axis=0)), axis=1),
                                       t[3:7, 3:7], equal_nan=True))

        self.assertTrue(np.array_equal(np.concatenate((np.concatenate((data[4:, 4:], data[:1, 4:]), axis=0),
                                                       np.concatenate((data[4:, :1], data[:1, :1]), axis=0)), axis=1), t[4:6, 4:6], equal_nan=True))
        self.assertTrue(np.array_equal(data[1:2, 1], t[1:2, 6], equal_nan=True))
        self.assertTrue(np.array_equal(np.concatenate((data[4:, 1], data[:2, 1]), axis=0), t[-1:2, 6], equal_nan=True))

if __name__ == '__main__':
    unittest.main()
