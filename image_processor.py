"""
The image processing library.
"""

import cv2 as cv
import numpy as np
from config import config


def process_image(img):
    """
    Process the current screenshot of the game and return the processed image.
    The process:
     - equalize the histogram to normalize the image
     - isolate the lane markings using morphological operations
     - find the edges using the Canny edge detector
     - blur the image with Gaussian blur
     - mask the image and retain only a specific region of interest
     - use the Hough probabilistic transform to find lines in image
     - keep only N longest lines
     - get the two lanes by averaging the lines
     - draw the lanes and lines on the image

    :param img: The current screenshot of the game to be processed (image)
    :return: the processed image with drawn lane, the lanes (array of length 4)
    """
    equ_img = equalize_hist(img)

    i_img = isolate_lane_markings(equ_img)

    edges = cv.Canny(i_img, config['canny_t1'], config['canny_t2'])

    blurred_img = cv.GaussianBlur(edges, (config['blur_sigma'], config['blur_sigma']), 0)
    masked_img = set_roi(blurred_img, np.int32(config['roi']))

    lines = hough_lines(masked_img)

    if lines is None:
        return masked_img, None, img

    lines = find_longest_lines(lines)

    lines = filter_lines(lines)

    lanes = find_lanes(lines)
    lane_lines = [make_line_points(720, 420, lanes[0]), make_line_points(720, 420, lanes[1])]

    img_lines = cv.cvtColor(masked_img, cv.COLOR_GRAY2BGR)
    img_lines = draw_lines(img_lines, lines, color=(200, 0, 0), thickness=2)
    img_lines = draw_lines(img_lines, lane_lines, color=(0, 0, 200), thickness=4)

    orig_img_lanes = draw_lines(img, lane_lines, color=(0, 0, 200), thickness=4)

    ret_lanes = (list(lanes[0]) if lanes[0] is not None else [-666, -666]) + \
                (list(lanes[1]) if lanes[1] is not None else [666, 666])

    return img_lines, ret_lanes, orig_img_lanes


def equalize_hist(img):
    """
    Normalize the image histogram to improve contrast and improve detection in low-light environments
    :param img: the image
    :return: the normalized image
    """

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    equ = cv.equalizeHist(gray)
    return gray


def isolate_lane_markings(img):
    """
    Isolate the lane markings with morphological operations.
    :param img: the image
    :return: the image with lane markings
    """

    kernel = np.ones((5, 5), np.uint8)
    opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel=kernel)
    dilation = cv.dilate(opening, kernel=kernel, iterations=1)

    return dilation


def find_longest_lines(lines):
    """
    Find the N longest lines.
    :param lines: the lines found by Hough transform
    :return: an array of length N, containing the N longest lines
    """

    # get lengths of all lines
    line_lengths = map(lambda l: np.sqrt((l[2] - l[0]) ** 2 + (l[3] - l[1]) ** 2), lines)

    # choose N longest lines
    longest_lines = np.array([lines[i] for i, _ in
                             sorted(enumerate(line_lengths), key=lambda p: p[1])[::-1][:config['n_longest_lines']]])

    return longest_lines


def filter_lines(lines):
    d = 100
    b = (550, 750)

    def f(line):
        x1, y1, x2, y2 = line
        return not(abs(x1 - x2) < d and b[0] < x1 < b[1] and b[0] < x2 < b[1])

    return np.array(list(filter(f, lines)))


def find_lanes(lines):
    """
    Determines two lanes from a set of lines.
    Separate the lines into two sets based on their slope, then use a weighted average of slopes / intercepts
    (the weights are line lengths) to determine lanes. If a lane cannot be found, return None in its place.
    Taken from https://github.com/naokishibuya/car-finding-lane-lines

    :param lines: The array of lines, each line is represented with four coordinates.
    :return: the left and right lane, each is represented with slope and intercept.
    """

    left_lines = []
    left_weights = []
    right_lines = []
    right_weights = []

    for line in lines:
        x1, y1, x2, y2 = line

        # ignore a vertical line
        if x2 == x1:
            continue

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

        # if 0.0

        if slope < 0:  # y is reversed in image
            left_lines.append((slope, intercept))
            left_weights.append(length)
        else:
            right_lines.append((slope, intercept))
            right_weights.append(length)

    # add more weight to longer lines
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None

    return left_lane, right_lane


def make_line_points(y1, y2, line):
    """
    Convert a line represented in slope and intercept into pixel points.
    Taken from https://github.com/naokishibuya/car-finding-lane-lines

    :param y1: first y coordinate of the line
    :param y2: second y coordinate of the line
    :param line: the slope and intercept of the line
    :return: an array of four coordinates [x1, y1, x2, y2]
    """

    if line is None:
        return None

    slope, intercept = line

    try:
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        y1 = int(y1)
        y2 = int(y2)
        return [x1, y1, x2, y2]

    except OverflowError:
        return None


def set_roi(img, roi):
    """
    Mask the image, retain specified region of interest.

    :param img: the image to be masked
    :param roi: the region of interest
    :return: masked image
    """

    mask = np.zeros_like(img)
    cv.fillConvexPoly(mask, roi, (255, 255, 255))
    return cv.bitwise_and(img, mask)


def hough_lines(edges):
    """
    Fit lines to the edges in the image using the Hough probabilistic transform.

    :param edges: the result of Canny's edge detector (blurred)
    :return: an array of lines found by Hough transform
    """

    lines = cv.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=config['hough_threshold'],
                           minLineLength=config['hough_min_line_length'], maxLineGap=config['hough_max_line_gap'])
    if lines is None:
        return None

    return np.array(list(map(lambda x: x[0], lines)))


def draw_lines(img, lines, color=(255, 255, 255), thickness=3):
    """
    Draw lines on the image.

    :param img: the image
    :param lines: the lines (each is an array of point coordinates)
    :param color: the color of the lines (default white)
    :param thickness: the thickness of the lines (default 3px)
    :return: the image with drawn lines
    """

    for line in lines:
        if line is None:
            continue
        x1, y1, x2, y2 = line
        cv.line(img, (x1, y1), (x2, y2), color, thickness)

    return img
