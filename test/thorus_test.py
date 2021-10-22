import unittest

import numpy as np

from src.Thorus import Thorus


class MyTestCase(unittest.TestCase):

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

    def test_get_1d(self):
        data = np.arange(2)
        t = Thorus.make(data)

        # print("T")
        # print("T",t)
        #
        # print("T2[0]")
        # print("T2[0]={}".format(t[0]))
        # print("T2[0:1]")
        # print("T2[0:1]={}".format(t[0:1]))
        # print("T2[3]")
        # print("T2[3]={}".format(t[3]))
        # print("T2[2]")
        # print("T2[2]={}".format(t[3]))
        # print("T2[1:3]")
        # print("T2[1:3]={}".format(t[1:3]))
        # print("T2[-1:1]")
        # print("T2[-1:1]={}".format(t[-1:1]))
        # print("T2[-1:0]")
        # print("T2[-1:0]={}".format(t[-1:0]))
        # print("T2[2:3]")
        # print("T2[2:3]={}".format(t[2:3]))

        self.assertTrue(np.array_equal(data[0], t[0], equal_nan=True))
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
        # print("T")
        # print("T", t)
        # #
        # print("T2[0]")
        # print("T2[0]={}".format(t[0]))
        # print("T2[0:1]")
        # print("T2[0:1]={}".format(t[0:1]))
        # print("T2[3]")
        # print("T2[3]={}".format(t[3]))
        # print("T2[2]")
        # print("T2[2]={}".format(t[3]))
        # print("T2[1:3]")
        # with self.assertRaises(Exception) as e:
        #     print(t[1:3])
        # self.assertTrue(type(e.exception) is ValueError)
        #
        # print("T2[-1:1]")
        # print("T2[-1:1]={}".format(t[-1:1]))
        # print("T2[-1:0]")
        # print("T2[-1:0]={}".format(t[-1:0]))

        # print("T2[2:3]")
        # with self.assertRaises(Exception) as e:
        #     print(t[2:3])
        # self.assertTrue(type(e.exception) is ValueError)

        self.assertTrue(np.array_equal(data[0], t[0], equal_nan=True))
        self.assertTrue(np.array_equal(data[0], t[2], equal_nan=True))
        self.assertTrue(np.array_equal(data[0, 0], t[0, 3], equal_nan=True))
        self.assertTrue(np.array_equal(data[-1], t[-1], equal_nan=True))
        self.assertTrue(np.array_equal(data[0:1], t[0:1], equal_nan=True))
        self.assertTrue(np.array_equal(data[-1:], t[-1:], equal_nan=True))
        self.assertTrue(np.array_equal(data[:], t[:], equal_nan=True))
        self.assertTrue(np.array_equal(data[1:3], t[1:3], equal_nan=True))
        self.assertTrue(np.array_equal(data[2:3], t[2:3], equal_nan=True))

        # # t[0:1, 0:3] = [1, 2, 3]
        # print("t", t)
        # print("\nt[0,0]", t[0, 0])
        # print(" \nt[0,1:3]", t[0, 1:3])
        # print(" \nt[0,2:4]", t[0, 2:4])
        # print("\n np.asarray([[1,2,3]])[0,-1:1]=", np.asarray([[0, 1, 2]])[0, -1:1],
        #       "\n np.asarray([[1,2,3]])[0,-1:0]=", np.asarray([[0, 1, 2]])[0, -1:0],
        #       "\n np.asarray([[1,2,3]])[0,1:2]=", np.asarray([[0, 1, 2]])[0, 1:2],
        #       "\n np.asarray([[1,2,3]])[0,4:8]=", np.asarray([[0, 1, 2]])[0, 4:8],
        #       "\n np.asarray([[1,2,3]])[0,-2:-1]=", np.asarray([[0, 1, 2]])[0, -2:-1],
        #       "\n np.asarray([[1,2,3]])[0,-5:-1]=", np.asarray([[0, 1, 2]])[0, -5:-1])

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

        # print("t", t)

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
