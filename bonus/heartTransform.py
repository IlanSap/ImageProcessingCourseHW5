import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def parametric_x(t):
    return np.sin(t) ** 3


def parametric_y(t):
    return 0.5 * np.cos(4 * t) + 2 * np.cos(3 * t) + 4 * np.cos(2 * t) - 13 * np.cos(t)


def find_hough_shape(image, edge_image, r_min, r_max, bin_threshold):
    img_height, img_width = image.shape[:2]

    # Note that cos and sin work with radians
    thetas = np.arange(0, 360, step=2)

    ## SIMPLE ##
    rs = np.arange(r_min, r_max, 0.2)

    ## MED ##
    # rs = np.arange(r_min, r_max, 0.4)

    ## HARD ##
    # rs = np.arange(r_min, r_max, 0.4)

    # Calculate Cos(theta) and Sin(theta), it will be required later
    cos_thetas = parametric_y(np.deg2rad(thetas))
    sin_thetas = parametric_x(np.deg2rad(thetas))

    # Quantize thetas and radii
    shape_candidates = []
    for r in rs:
        for theta in thetas:
            shape_candidates.append((r, theta))

    # Hough Accumulator. We are using defaultdict instead of standard dict as this will initialize for keys which are not already present in the dictionary instead of throwing an exception.
    accumulator = defaultdict(int)

    edge_points = np.argwhere(edge_image > 0)
    for point in edge_points:
        y, x = point
        for r, theta in shape_candidates:
            # Your code goes here
            # Don't forget - the points should be integers!
            x_center = int(x - (14.5 * r * (np.sin(np.deg2rad(theta)) ** 3)))
            y_center = int(y - r * (0.5 * (np.cos(np.deg2rad(4 * theta))) + 2 * (np.cos(np.deg2rad(3 * theta))) + 4 * (
                np.cos(np.deg2rad(2 * theta))) - 13 * (np.cos(np.deg2rad(theta)))))
            # vote for current candidate\
            accumulator[(x_center, y_center, r)] += 1

    # Output image with detected lines drawn
    output_img = image.copy()
    # Output list of detected circles. A single circle would be a tuple of (x,y,r,threshold)
    out_shapes = []
    # Sort the accumulator based on the votes for the candidate circles
    for candidate_shape, votes in sorted(accumulator.items(), key=lambda i: -i[1]):
        x, y, r = candidate_shape
        num_thetas = 180
        current_vote_percentage = votes / num_thetas
        if (current_vote_percentage > bin_threshold):
            out_shapes.append((x, y, r, current_vote_percentage))
    # If the current_vote_percentage is larger than bin_threshold then we want that shape. append x,y,r and the precentage to the out_shapes.

    # DO NOT EDIT
    pixel_threshold = 10
    postprocess_shapes = []
    for x, y, r, v in out_shapes:
        # Exclude shapes that are too close of each other
        # Remove nearby duplicate circles based on postprocess_shapes
        if all(abs(x - xc) > pixel_threshold or abs(y - yc) > pixel_threshold > pixel_threshold and abs(
                r - rc) > pixel_threshold for xc, yc, rc, v in postprocess_shapes):
            postprocess_shapes.append((x, y, r, v))
    out_shapes = postprocess_shapes

    # Draw shortlisted hearts on the output image
    for x, y, r, v in out_shapes:
        # We found heart at (x,y) with radius r.
        # Generate all the x1 and y1 points of that heart.
        x1 = x + 14.5 * r * sin_thetas
        y1 = y + r * cos_thetas

        colors = [
            (255, 0, 0),  # 'b' (blue)
            (0, 255, 0),  # 'g' (green)
            (255, 255, 0),  # 'c' (cyan)
            (255, 0, 255),  # 'm' (magenta)
            (0, 255, 255),  # 'y' (yellow)
            (0, 0, 0),  # 'k' (black)
        ]

        color_chars = ['b', 'g', 'c', 'm', 'y', 'k']

        id = np.random.randint(len(colors))
        color1 = colors[id]
        color2 = color_chars[id]

        plt.plot(x1, y1, markersize=1.5, color=color2)
        output_img = cv2.circle(output_img, (x, y), 1, color1, -1)
        print(x, y, r, v)

    return output_img


## SIMPLE ##
IMAGE_NAME = "simple"

## MED ##
# IMAGE_NAME = "med"

## HARD ##
# IMAGE_NAME = "hard"


image = cv2.imread(f'{IMAGE_NAME}.jpg')
edge_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

min_edge_threshold, max_edge_threshold = 100, 200
# Apply edge detector with min_edge_threshold, max_edge_threshold
edge_image = cv2.Canny(edge_image, min_edge_threshold, max_edge_threshold)

## SIMPLE ##
r_min = 5
r_max = 8
bin_threshold = 0.28

## MED ##
# r_min = 2
# r_max = 9
# bin_threshold = 0.28

## HARD ##
# r_min = 2
# r_max = 12
# bin_threshold = 0.28

if edge_image is not None:

    print("Attempting to detect Hough hearts...")
    results_img = find_hough_shape(image, edge_image, r_min, r_max, bin_threshold)

    if results_img is not None:
        plt.imshow(cv2.cvtColor(results_img, cv2.COLOR_BGR2RGB))
        plt.show()
        cv2.imwrite(f'{IMAGE_NAME}_detected.jpg', results_img)
    else:
        print("Error in input image!")

    print("Hough hearts detection complete!")