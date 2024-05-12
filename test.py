import unittest

import main

class Truncate(unittest.TestCase):
    def test_truncate(self):
        self.assertEqual(main.truncate(8, 1), 4)
        self.assertEqual(main.truncate(8, 2), 2)
        self.assertEqual(main.truncate(9, 2), 2)

if __name__ == '__main__':
    unittest.main()