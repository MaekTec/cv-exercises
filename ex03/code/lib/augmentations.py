import torch
from PIL import Image
import numpy as np

class horizontal_flip(torch.nn.Module):
    """
    Flip the image along the second dimension with a probability of p
    """

    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, img):
        # START TODO #################
        # convert the image to numpy
        # draw a random number
        # flip the image in the second dimension 
        # if this number is smaller than self.p
        img = np.asarray(img) # (height, width, 3) uint8
        random_number = np.random.random_sample()
        if random_number < self.p:
            img = img[:, ::-1, :]
        return img
        # END TODO #################


class random_resize_crop(torch.nn.Module):
    """
    simplified version of resize crop, which keeps the aspect ratio of the image.
    """
    def __init__(self, size, scale):
        """ initialize this transform
        Args:
            size (int): size of the image
            scale (tuple(int)): upper and lower bound for resizing image"""
        super().__init__()
        self.size = size
        self.scale = scale


    def _uniform_rand(self, low, high):
        return np.random.random_sample() * (high - low) + low

    def forward(self, img):
        # START TODO #################
        height, width, _ = img.shape
        assert height == width
        input_size = height

        crop_size = int(self._uniform_rand(self.scale[0], self.scale[1]) * input_size)
        x_start = np.random.randint(0, input_size-crop_size)
        y_start = np.random.randint(0, input_size-crop_size)
        crop = img[x_start:x_start+input_size, y_start:y_start+input_size]

        img = Image.fromarray(img) 
        img = img.resize((self.size, self.size), resample=Image.BILINEAR)
        return img
        # END TODO #################

