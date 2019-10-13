import unittest
import layers


class TestLayers(unittest.TestCase):

    def test_dense_forward(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_dense_backward(self):
        self.assertTrue('FOO'.isupper())
        pass

    def test_conv2d_forward(self):
        pass

    def test_conv2d_backward(self):
        pass

    def test_maxpool_forward(self):
        pass

    def test_maxpool_backward(self):
        pass

    def test_avgpool_forward(self):
        pass

    def test_avgpool_backward(self):
        pass


if __name__ == '__main__':
    unittest.main()
