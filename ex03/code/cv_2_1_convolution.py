import cv2 
import numpy as np

def convolve2D(image, kernel, padding=0, strides=1):

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    # START TODO ###################
    # xOutput =
    # yOutput = 
    raise NotImplementedError
    # END TODO ###################
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        # START TODO ###################
        # imagePadded = 
        raise NotImplementedError
        # END TODO ###################
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        # START TODO ###################
        raise NotImplementedError
        # END TODO ###################
        
        # Only Convolve if y has gone down by the specified Strides
        # START TODO ###################
        raise NotImplementedError
        # END TODO ###################

    return output


if __name__ == '__main__':
    # Grayscale Image
    image = cv2.imread('image.png',0)

    # Edge Detection Kernel
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    # Convolve and Save Output
    output = convolve2D(image, kernel, padding=2)
    cv2.imwrite('2DConvolved.png', output)
