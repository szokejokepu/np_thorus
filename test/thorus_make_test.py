import unittest

import numpy as np

from src.Thorus import Thorus


class MakeTestCase(unittest.TestCase):

    def test_make(self):
        data = [0]
        t = Thorus.make(data)
        self.assertTrue(np.array_equal(data, t, equal_nan=True), "simple array creation fails")

        data = np.arange(2)
        t = Thorus.make(data)
        self.assertTrue(np.array_equal(data, t, equal_nan=True), "simple array creation fails")

        data = np.arange(2, dtype=np.int)
        t = Thorus.make(data)
        self.assertTrue(np.array_equal(data, t, equal_nan=True), "dtype is not correctly inferred")

        data = np.arange(10).reshape((2, 5))
        t = Thorus.make(data)
        self.assertTrue(np.array_equal(data, t, equal_nan=True), "Complex shaped data fails")

if __name__ == '__main__':
    unittest.main()
