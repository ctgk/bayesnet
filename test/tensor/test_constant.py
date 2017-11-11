import unittest
from bayesnet import Constant


class TestConstant(unittest.TestCase):

    def test_constant(self):
        try:
            Constant(1)
        except Exception:
            self.fail()


if __name__ == '__main__':
    unittest.main()
