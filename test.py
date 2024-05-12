import unittest
import main

import numpy as np

class Truncate(unittest.TestCase):
    def test_truncate(self):
        self.assertEqual(main.truncate(8, 1), 4)
        self.assertEqual(main.truncate(8, 2), 2)
        self.assertEqual(main.truncate(9, 2), 2)

        self.assertEqual(main.truncate(42, 0), 42)
        self.assertEqual(main.truncate(123, 1), 61)
        self.assertEqual(main.truncate(123, 3), 15)
    
    def test_truncate_pixel(self):
        self.assertEqual(main.truncate_pixel(np.array([123, 8, 0])), 30784)
        self.assertEqual(main.truncate_pixel(np.array([255, 255, 255])), 65535)
        self.assertEqual(main.truncate_pixel(np.array([0, 0, 0])), 0)

    def test_retrieve_pixel(self):
        self.assertEqual(main.detruncate_pixel(30784).tolist(), [120, 8, 0])
        self.assertEqual(main.detruncate_pixel(65535).tolist(), [248, 252, 248])
        self.assertEqual(main.detruncate_pixel(0).tolist(), [0, 0, 0])

class Padding(unittest.TestCase):
    def test_shape(self):
        self.assertEqual(main.padded_shape((8, 8, 3)), (8, 8, 3))
        self.assertEqual(main.padded_shape((7, 8, 3)), (8, 8, 3))
        self.assertEqual(main.padded_shape((6, 9, 3)), (8, 12, 3))

if __name__ == '__main__':
    unittest.main()