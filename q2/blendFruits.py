import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift

NUM_OF_LEVELS = 5

def show_pyramid(pyramid):
    for image in pyramid:
        plt.imshow(image, cmap='gray')
        plt.axis('off')  # Turn off axis labels
        # Set the aspect ratio to 'auto' to prevent distortion of the images

        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()


def pyrUp(image, height, width):
    # Upsample the image using bilinear interpolation to resize it
    upsampled_image = cv2.resize(image, (height, width), interpolation=cv2.INTER_LINEAR)

    # Making the image more smooth
    smoothed_image = cv2.GaussianBlur(upsampled_image, (9, 9), sigmaX=2)

    return smoothed_image

def pyrDown(image):
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, (5, 5), sigmaX=2)

    # Downsample the image by taking every second pixel
    down_image = blurred_image[::2, ::2]
    return down_image

def build_gaussian_pyramid(image, levels):
    # insert to the first place in the pyramid the original image
    pyramid = [image]
    for _ in range(levels - 1):
        # applies Gaussian blurring and downsamples the image by a factor of 2 in both dimensions.
        image = pyrDown(image)
        pyramid.append(image)
    #show_pyramid(pyramid)

    return pyramid

def get_laplacian_pyramid(image, levels, resize_ratio=0.5):

    gaussian_pyramid = build_gaussian_pyramid(image, levels)

    laplacian_pyramid = []
    for i in range(levels -1):
        height, width = gaussian_pyramid[i].shape
        resize_next_image = pyrUp(gaussian_pyramid[i+1], height, width)

        # Subtract the upsampled image from the current image
        laplacian_image = gaussian_pyramid[i].astype(np.float32) - resize_next_image.astype(np.float32)
        laplacian_pyramid.append(laplacian_image)

    # insert the lase level in the gaussian pyramid to the Laplacian pyramid
    laplacian_pyramid.append(gaussian_pyramid[-1])
    # show_pyramid(laplacian_pyramid)

    return laplacian_pyramid

def restore_from_pyramid(pyramidList, resize_ratio=2):
    # save the last image in the pyramid (the smallest one)
    image = pyramidList[-1]
    for i in range(len(pyramidList) - 2, -1, -1):
        height, width = pyramidList[i].shape[1], pyramidList[i].shape[0]
        resize_image = pyrUp(image, height, width)

        image = resize_image + pyramidList[i]

    return image


def validate_operation(img):
    pyr = get_laplacian_pyramid(img, 5)
    restored_image = restore_from_pyramid(pyr)

    plt.title(f"MSE is {np.mean((restored_image - img) ** 2)}")
    plt.imshow(restored_image, cmap='gray')

    plt.show()


def create_mask(orange_level, cur_level):
    # Mask initialize
    mask = np.zeros_like(orange_level, np.float32)
    width = orange_level.shape[1]

    # Mask columns intialization to 1.0 as instructed.
    mask_index = int(0.5 * width - cur_level)
    mask[:, :mask_index] = 1.0

    for i in range(2 * (cur_level + 1)):
        mask[:, width // 2 - (cur_level + 1) + i] = 0.9 - 0.9 * i / (2 * (cur_level + 1))

    return mask


def blend_pyramids(pyr_apple, pyr_orange):
    blended_pyr = []

    for current_level in range(NUM_OF_LEVELS):
        apple_level = pyr_apple[current_level]
        orange_level = pyr_orange[current_level]

        mask = create_mask(orange_level, current_level)

        # Blend the two images in the current level into one
        pyr_level_blend = orange_level * mask + apple_level * (1 - mask)
        blended_pyr.append(pyr_level_blend)

    return blended_pyr


# main
apple = cv2.imread('apple.jpg')
apple = cv2.cvtColor(apple, cv2.COLOR_BGR2GRAY)

orange = cv2.imread('orange.jpg')
orange = cv2.cvtColor(orange, cv2.COLOR_BGR2GRAY)

# validate_operation(apple)
# validate_operation(orange)

pyr_apple = get_laplacian_pyramid(apple, NUM_OF_LEVELS)
pyr_orange = get_laplacian_pyramid(orange, NUM_OF_LEVELS)

pyr_result = blend_pyramids(pyr_apple, pyr_orange)
final = restore_from_pyramid(pyr_result)
plt.imshow(final, cmap='gray')
plt.show()
cv2.imwrite("result.jpg", final)

