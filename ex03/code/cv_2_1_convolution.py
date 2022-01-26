import cv2 
import numpy as np

def convolve2D(image, kernel, padding=0, strides=1):
    """
        taken from
        https://medium.com/analytics-vidhya/2d-convolution-using-python-numpy-43442ff5f381
        and fixed the indexing for the output image
    """
    # Do convolution instead of Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    xKernLeft = xKernShape // 2
    # since slicing end is exclusive, uneven kernel shapes would be too small
    xKernRight = int(np.around(xKernShape / 2.))
    yKernShape = kernel.shape[1]
    yKernUp = yKernShape // 2
    yKernDown = int(np.around(yKernShape / 2.))
    xImgShape = image.shape[1]
    yImgShape = image.shape[0]

    # Shape of Output Convolution
    # START TODO ###################
    xOutput = (xImgShape + 2*padding - xKernShape) // strides + 1
    yOutput = (yImgShape + 2*padding - yKernShape) // strides + 1
    
    # END TODO ###################
    output = np.zeros((yOutput, xOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        # START TODO ###################
        imagePadded = np.pad(image, padding)
        # END TODO ###################
    else:
        imagePadded = image

    # Indices for output image
    x_out = y_out = -1
    # Iterate through image
    for y in range(output.shape[1]):
        # START TODO ###################
        for x in range(output.shape[0]):
            # flip kernel to get convolution instead of cross-correlation
            output[x, y] = np.sum(imagePadded[x*strides:x*strides + xKernShape, y*strides:y*strides + yKernShape] * kernel[::-1, ::-1])
        # END TODO ###################
    return output


if __name__ == '__main__':
    # Grayscale Image
    image = cv2.imread('panda.jpeg', 0)
    print(image.shape)

    # Edge Detection Kernel
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    # Convolve and Save Output
    output = convolve2D(image, kernel, padding=2, strides=1)
    cv2.imwrite('2DConvolved.png', output)
