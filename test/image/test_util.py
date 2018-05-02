import unittest
import numpy as np
from bayesnet.image.util import img2patch, patch2img


class TestImg2Patch(unittest.TestCase):

    def test_img2patch(self):
        img = np.arange(16).reshape(1, 4, 4, 1)
        patch = img2patch(img, size=3, step=1)
        expected = np.asarray([
            [img[0, 0:3, 0:3, 0], img[0, 0:3, 1:4, 0]],
            [img[0, 1:4, 0:3, 0], img[0, 1:4, 1:4, 0]]
        ])
        expected = expected[None, ..., None]
        self.assertTrue((patch == expected).all())

        imgs = [
            np.random.randn(2, 5, 6, 3),
            np.random.randn(3, 10, 10, 2),
            np.random.randn(1, 23, 17, 5)
        ]
        sizes = [
            (1, 1),
            2,
            (3, 4)
        ]
        steps = [
            (1, 2),
            (3, 1),
            3
        ]
        shapes = [
            (2, 5, 3, 1, 1, 3),
            (3, 3, 9, 2, 2, 2),
            (1, 7, 5, 3, 4, 5)
        ]
        for img, size, step, shape in zip(imgs, sizes, steps, shapes):
            self.assertEqual(shape, img2patch(img, size, step).shape)


class TestPatch2Img(unittest.TestCase):

    def test_patch2img(self):
        img = np.arange(16).reshape(1, 4, 4, 1)
        patch = img2patch(img, size=2, step=2)
        self.assertTrue((img == patch2img(patch, (2, 2), (1, 4, 4, 1))).all())
        patch = img2patch(img, size=3, step=1)
        expected = np.arange(0, 32, 2).reshape(1, 4, 4, 1)
        expected[0, 0, 0, 0] /= 2
        expected[0, 0, -1, 0] /= 2
        expected[0, -1, 0, 0] /= 2
        expected[0, -1, -1, 0] /= 2
        expected[0, 1:3, 1:3, 0] *= 2
        self.assertTrue((expected == patch2img(patch, (1, 1), (1, 4, 4, 1))).all())


if __name__ == '__main__':
    unittest.main()
