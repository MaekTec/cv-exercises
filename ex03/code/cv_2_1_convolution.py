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
    xOutput = (xImgShape + 2*padding - xKernShape) // strides + 1
    yOutput = (yImgShape + 2*padding - yKernShape) // strides + 1
    
    # END TODO ###################
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        # START TODO ###################
        imagePadded = np.pad(image, padding)
        # END TODO ###################
    else:
        imagePadded = image

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
