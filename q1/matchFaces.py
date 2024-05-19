import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from sklearn.cluster import DBSCAN

import warnings

warnings.filterwarnings("ignore")


# Implementation of the 'scale_down' function as per the assignment instructions.
# Scales down an image by a given ratio using the Fourier transform method.
def scale_down(image, resize_ratio):
    # Apply Fourier transform
    F = fftshift(fft2(image))
    # Calculate the new size after applying the resize_ratio
    new_size = np.array(F.shape) * resize_ratio
    new_size = new_size.astype(int)  # Convert to integer size
    # Extract the center part of the frequency domain according to the new size
    center_indices = np.array(F.shape) // 2
    cropped_F = F[center_indices[0] - new_size[0] // 2:center_indices[0] + new_size[0] // 2,
                center_indices[1] - new_size[1] // 2:center_indices[1] + new_size[1] // 2]
    # Apply inverse Fourier transform to get the scaled image
    scaled_image = ifft2(ifftshift(cropped_F))
    return np.abs(scaled_image)


# Implementation of the 'scale_up' function as per the assignment instructions.
# Scales up an image by a given ratio using the Fourier transform method.
def scale_up(image, resize_ratio):
    # Apply Fourier transform
    F = fftshift(fft2(image))
    # Calculate padding size
    pad_size = ((np.array(F.shape) * (resize_ratio - 1)) / 2).astype(int)
    # Zero padding in the frequency domain
    padded_F = np.pad(F, ((pad_size[0], pad_size[0]), (pad_size[1], pad_size[1])), mode='constant')
    # Apply inverse Fourier transform to get the scaled image
    scaled_image = ifft2(ifftshift(padded_F))
    return np.abs(scaled_image)


# Implementation of the 'ncc_2d' function as per the assignment instructions.
# Calculates the normalized cross-correlation (NCC) map between an image and a pattern.
def ncc_2d(image, pattern):
    # Convert the pattern to float
    pattern = pattern.astype(np.float64)
    # Calculate the dimensions of the NCC map
    ncc_map_shape = (image.shape[0] - pattern.shape[0] + 1, image.shape[1] - pattern.shape[1] + 1)
    ncc_map = np.zeros(ncc_map_shape)

    # Compute mean and standard deviation of the pattern
    pattern_mean = np.mean(pattern)
    pattern_std = np.std(pattern)

    # Normalizing the pattern
    pattern = (pattern - pattern_mean) / (pattern_std if pattern_std else 1)

    # Perform NCC calculation
    for i in range(ncc_map_shape[0]):
        for j in range(ncc_map_shape[1]):
            # Extract the current window from the image
            window = image[i:i + pattern.shape[0], j:j + pattern.shape[1]]

            # Compute mean and standard deviation of the window
            window_mean = np.mean(window)
            window_std = np.std(window)

            # If the window is not a constant value (std != 0), compute NCC
            if window_std != 0:
                ncc_value = np.sum((window - window_mean) * pattern) / (window_std * window.size)
                ncc_map[i, j] = ncc_value

    return ncc_map


# Displays an image, pattern, and their NCC heatmap side by side.
def display(image, pattern):
    plt.subplot(2, 3, 1)
    plt.title('Image')
    plt.imshow(image, cmap='gray')

    plt.subplot(2, 3, 3)
    plt.title('Pattern')
    plt.imshow(pattern, cmap='gray', aspect='equal')

    ncc = ncc_2d(image, pattern)

    plt.subplot(2, 3, 5)
    plt.title('Normalized Cross-Correlation Heatmap')
    plt.imshow(ncc ** 2, cmap='coolwarm', vmin=0, vmax=1, aspect='auto')

    cbar = plt.colorbar()
    cbar.set_label('NCC Values')

    plt.show()


# Draws rectangles around detected matches in an image.
def draw_matches(image, matches, pattern_size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for point in matches:
        y, x = point
        top_left = (int(x - pattern_size[1] / 2), int(y - pattern_size[0] / 2))
        bottom_right = (int(x + pattern_size[1] / 2), int(y + pattern_size[0] / 2))
        cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 1)

    plt.imshow(image, cmap='gray')
    plt.show()

    cv2.imwrite(f"{CURR_IMAGE}_result.jpg", image)


# Clustering matches to ensure each face is marked only once
def cluster_matches(matches, eps=10, min_samples=1):
    # Use DBSCAN to cluster close points
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(matches)
    unique_matches = []
    for cluster_id in np.unique(clustering.labels_):
        if cluster_id != -1:  # Ignore noise points
            # Calculate the centroid of the points in each cluster
            points = matches[clustering.labels_ == cluster_id]
            centroid = np.mean(points, axis=0).astype(int)
            unique_matches.append(centroid)
    return np.array(unique_matches)


CURR_IMAGE = "students"

image = cv2.imread(f'{CURR_IMAGE}.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

pattern = cv2.imread('template.jpg')
pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)

############# DEMO #############
display(image, pattern)


############# Students #############

image_scaled = image
pattern_scaled = scale_down(pattern, 0.5)

display(image_scaled, pattern_scaled)

ncc = ncc_2d(image_scaled, pattern_scaled)
real_matches = np.argwhere(ncc >= 0.55)

real_matches[:, 0] += pattern_scaled.shape[0] // 2
real_matches[:, 1] += pattern_scaled.shape[1] // 2

unique_matches = cluster_matches(real_matches)
draw_matches(image, unique_matches, pattern_scaled.shape)


############# Crew #############

CURR_IMAGE = "thecrew"

image = cv2.imread(f'{CURR_IMAGE}.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

pattern = cv2.imread('template.jpg')
pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)

image_scaled = scale_up(image, 2)
pattern_scaled = scale_down(pattern, 0.5)

display(image_scaled, pattern_scaled)

ncc = ncc_2d(image_scaled, pattern_scaled)
real_matches = np.argwhere(ncc >= 0.44)

real_matches[:, 0] += pattern_scaled.shape[0] // 2
real_matches[:, 1] += pattern_scaled.shape[1] // 2

real_matches = real_matches / 2

unique_matches = cluster_matches(real_matches)
draw_matches(image, unique_matches, pattern_scaled.shape)
