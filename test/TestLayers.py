import unittest
import layers


class TestLayers(unittest.TestCase):

    def test_dense_forward(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_dense_backward(self):
        self.assertTrue('FOO'.isupper())
        pass

    def test_conv2d_forward(self):
        img_input = [[260.745, 261.332, 112.27, 262.351],
                     [260.302, 208.802, 139.05, 230.709],
                     [261.775, 93.73, 166.118, 122.847],
                     [259.56, 232.038, 262.351, 228.937]]

        filter = [[1, 0],
                  [0, 1]]

        conv_layer = layers.Conv2D()

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
