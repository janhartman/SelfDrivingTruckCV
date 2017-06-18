import cv2 as cv
import numpy as np

"""
The configuration for image processing.
"""
config = {
    'roi': np.int32([[50, 720], [50, 450], [550, 350], [730, 350], [1210, 450], [1210, 720]]),
    'perspective_roi': np.float32([[0, 720], [600, 350], [680, 350], [1280, 720]]),
    'canny_t1': 100,
    'canny_t2': 250,
    'blur_sigma': 3,
    'hough_threshold': 10,
    'hough_min_line_length': 50,
    'hough_max_line_gap': 30,
    'n_longest_lines': 10,
}


def process_image(img):
    """
    Process the current screenshot of the game and return the processed image.
    The current process is:
     - find the edges using the Canny edge detector
     - mask the image and retain only a specific region of interest
     - blur the image with Gaussian blur
     - use the Hough probabilistic transform to find lines in image
     - find lanes
     - draw the lanes on the image

    :param img: The current screenshot of the game to be processed
    :return: the processed image with drawn lanes and the lanes
    """
    i_img = isolate_lane_markings(img)

    edges = cv.Canny(i_img, config['canny_t1'], config['canny_t2'])

    blurred_img = cv.GaussianBlur(edges, (config['blur_sigma'], config['blur_sigma']), 0)
    masked_img = set_roi(blurred_img, config['roi'])

    lines = hough_lines(masked_img)

    if lines is None:
        return masked_img, None

    lines = find_longest_lines(lines)

    lanes = average_slope_intercept(lines)
    lane_lines = [make_line_points(720, 420, lanes[0]), make_line_points(720, 420, lanes[1])]

    if None in lane_lines:
        return masked_img, None

    img_lines = cv.cvtColor(masked_img, cv.COLOR_GRAY2BGR)
    img_lines = draw_lines(img_lines, lines, color=(200, 0, 0), thickness=2)
    img_lines = draw_lines(img_lines, lane_lines, color=(0, 0, 200), thickness=4)

    return img_lines, list(lanes[0]) + list(lanes[1])


def isolate_lane_markings(img):
    """
    Isolate the lane markings with morphological operations.
    :param img: the image
    :return: the image with lane markings
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel=kernel)
    dilation = cv.dilate(opening, kernel=kernel, iterations=1)

    return dilation


def find_longest_lines(lines):
    """
    Find the N longest lines.
    :param lines: the lines found by Hough transform
    :return: an array of longest lines
    """

    # get lengths of all lines
    line_lengths = map(lambda l: np.sqrt((l[2] - l[0]) ** 2 + (l[3] - l[1]) ** 2), lines)

    # choose N longest lines
    longest_lines = np.array([lines[i] for i, _ in
                             sorted(enumerate(line_lengths), key=lambda p: p[1])[::-1][:config['n_longest_lines']]])

    return longest_lines


def average_slope_intercept(lines):
    left_lines = []  # (slope, intercept)
    left_weights = []  # (length,)
    right_lines = []  # (slope, intercept)
    right_weights = []  # (length,)

    for line in lines:
        x1, y1, x2, y2 = line
        if x2 == x1:
            continue  # ignore a vertical line
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
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
    Convert a line represented in slope and intercept into pixel points
    """
    if line is None:
        return None

    slope, intercept = line

    # make sure everything is integer as cv2.line requires it
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
    masked = cv.bitwise_and(img, mask)
    return masked


def hough_lines(edges):
    """
    Fit lines to the edges in the image.

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
    :param lines: the lines (provided as a list of point coordinates)
    :param color: the color of the lines (default white)
    :param thickness: the thickness of the lines (default 3px)
    :return: the image with drawn lines
    """

    for line in lines:
        x1, y1, x2, y2 = line
        cv.line(img, (x1, y1), (x2, y2), color, thickness)

    return img
