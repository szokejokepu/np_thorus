import unittest

import numpy as np

from src.Thorus import Thorus


def get_random_numbers_list(shape):
    if isinstance(shape, int):
        random_numbers = np.random.rand(shape)
    else:
        random_numbers = np.random.rand(*shape)
    return random_numbers, random_numbers


class SetTestCase(unittest.TestCase):

    def test_set_1d(self):
        data = np.arange(2)
        t = Thorus.make(data)

        data[0], t[0] = get_random_numbers_list(1)
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))
        data[0], t[2] = get_random_numbers_list(1)
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))
        data[1], t[3] = get_random_numbers_list(1)
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))
        data[-1], t[-1] = get_random_numbers_list(1)
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))
        data[-2], t[-2] = get_random_numbers_list(1)
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))

        data[0:1], t[0:1] = get_random_numbers_list(1)
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))
        data[1:2], t[1:2] = get_random_numbers_list(1)
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))
        data[1:2], t[-1:0] = get_random_numbers_list(1)
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))
        data[0:1], t[2:3] = get_random_numbers_list(1)
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))

        data[0:2], t[0:2] = get_random_numbers_list(2)
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))
        data[:], t[:] = get_random_numbers_list(2)
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))

        data2, t[1:3] = get_random_numbers_list(2)
        data[1:2], data[0:1] = data2
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))

        data2, t[-1:1] = get_random_numbers_list(2)
        data[1:2], data[0:1] = data2
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))

        data[0:], t[0:] = get_random_numbers_list(2)
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))
        data[:1], t[:1] = get_random_numbers_list(1)
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))

        data2, t[:3] = get_random_numbers_list(3)
        data[:], data[0:1] = data2[:2], data2[2:]
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))

    def test_set_1d_transposed(self):
        data = np.arange(3).reshape((1, 3))
        t = Thorus.make(data)

        data[0], t[0] = get_random_numbers_list(3)
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))
        data[0], t[2] = get_random_numbers_list(3)
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))
        data[-1], t[-1] = get_random_numbers_list(3)
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))

        data[0:1], t[0:1] = get_random_numbers_list(1)
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))

        data[0, 0], t[0, 3] = get_random_numbers_list(1)
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))

        data[:], t[:] = get_random_numbers_list(3)
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))

        data[2:3], t[2:3] = get_random_numbers_list(3)
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))
        data[1:3], t[1:3] = get_random_numbers_list(3)
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))

        data2, t[0, -1:1] = get_random_numbers_list(2)
        data[0, -1:], data[0, 0:1] = data2
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))

        data[0, -1:], t[0, -1:] = get_random_numbers_list(1)
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))

        data[0, 1:2], t[0, 1:2] = get_random_numbers_list(1)
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))

        data[0, 4:8], t[0, 4:8] = get_random_numbers_list(0)
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))

        data[0, -2:-1], t[0, -2:-1] = get_random_numbers_list(1)
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))

    def test_set_2d(self):
        data = np.arange(25).reshape((5, 5))
        t = Thorus.make(data)

        data[0, 0], t[0, 0] = get_random_numbers_list(1)
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))
        data[0, 0], t[0, 5] = get_random_numbers_list(1)
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))
        data[0, 0], t[5, 0] = get_random_numbers_list(1)
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))
        data[0, 0], t[5, 5] = get_random_numbers_list(1)
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))
        data[-1, -1], t[-1, -1] = get_random_numbers_list(1)
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))
        data[-1, -1], t[4, 4] = get_random_numbers_list(1)
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))
        data[4, 4], t[-1, -1] = get_random_numbers_list(1)
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))

        data[:, :], t[:, :] = get_random_numbers_list((5, 5))
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))
        data[3:, 3:], t[3:, 3:] = get_random_numbers_list((2, 2))
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))

        data2, t[0, :7] = get_random_numbers_list(7)
        data[0, :], data[0, 0:2] = data2[:5], data2[5:]
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))

        data2, t[:7, 0] = get_random_numbers_list(7)
        data[:, 0], data[0:2, 0] = data2[:5], data2[5:]
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))

        data2, t[0:1, :7] = get_random_numbers_list((1, 7))
        data[0:1, :], data[0:1, 0:2] = data2[0:1, :5], data2[0:1, 5:]
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))

        data2, t[:7, 0:1] = get_random_numbers_list((7, 1))
        data[:, 0:1], data[0:2, 0:1] = data2[:5, 0:1], data2[5:, 0:1]
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))

        data2, t[0:1, 3:7] = get_random_numbers_list((1, 4))
        data[0:1, 3:], data[0:1, :2] = data2[0:1, :2], data2[0:1, 2:]
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))

        data2, t[3:7, 0:1] = get_random_numbers_list((4, 1))
        data[3:, 0:1], data[:2, 0:1] = data2[:2, 0:1], data2[2:, 0:1]
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))

        data2, t[:, 3:7] = get_random_numbers_list((5, 4))
        data[:, 3:], data[:, :2] = data2[:, :2], data2[:, 2:]
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))

        data2, t[3:7, :] = get_random_numbers_list((4, 5))
        data[3:, :], data[:2, :] = data2[:2, :], data2[2:, :]
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))

        data2, t[3:7, 3:7] = get_random_numbers_list((4, 4))
        data[3:5, 3:5], data[0:2, 3:5], data[3:5, 0:2], data[0:2, 0:2] = data2[0:2, 0:2], data2[2:, 0:2], \
                                                                         data2[0:2,2:], data2[2:,2:]
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))

        data2, t[4:6, 4:6] = get_random_numbers_list((2, 2))
        data[4:5, 4:5], data[0:1, 4:5], data[4:5, 0:1], data[0:1, 0:1] = data2[0:1, 0:1], data2[1:, 0:1], \
                                                                         data2[0:1,1:], data2[1:,1:]
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))

        data[1:2, 1], t[1:2, 6] = get_random_numbers_list((1,1))
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))

        data2, t[-1:2, 6] = get_random_numbers_list(3)
        data[4:, 1], data[:2, 1] = data2[:1], data2[1:]
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))


if __name__ == '__main__':
    unittest.main()
