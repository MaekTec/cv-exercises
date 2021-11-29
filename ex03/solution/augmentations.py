import torch
from PIL import Image
import numpy as np

class horizontal_flip(torch.nn.Module):
    """

    """

    def __init__(self, p):
        # todo by sutdents
        super().__init__()
        self.p = p

    def forward(self, img):
        #convert to numpy. This is slower than using PIL but more instructional.
        img = np.array(img)
        if torch.rand(1) < self.p:
            img = img[:, ::-1, :]
        return Image.fromarray(img)


class random_resize_crop(torch.nn.Module):
    """
    simplified version of resize crop, which keeps the aspect ratio of the image.
    """
    def __init__(self, size, scale):
        super().__init__()
        self.size = size
        self.scale = scale


    def _uniform_rand(self, low, high):
        return np.random.rand(1)[0] * (low - high) + high

    def forward(self, img):
        # todo by sutdents
        # resize the image
        scale = self._uniform_rand(self.scale[0], self.scale[1])
        w, h = img.size
        new_size = np.array(np.round((w * scale, h * scale)), dtype=int)
        img = img.resize(new_size, resample=Image.BILINEAR)

        #again we cast to numpy but using PIL would be faster
        img = np.array(img)
        #crop
        max_top_left = (new_size[0] - w, new_size[1] - h)
        top_left = (self._uniform_rand(0, max_top_left[0]),
                    self._uniform_rand(0, max_top_left[1]))
        bottom_right = (top_left[0] + self.size,
                        top_left[1] + self.size)

        # round and cast to int
        top_left = np.array(np.round(top_left), dtype=int)
        bottom_right = np.array(np.round(bottom_right), dtype=int)

        crop = img[top_left[0]:bottom_right[0],
                   top_left[1]:bottom_right[1],
                   :]
        return Image.fromarray(crop)

