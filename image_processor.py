import cv2 as cv
import numpy as np

"""
The configuration for image processing.
"""
config = {
    'roi_higher': np.array([[0, 720], [0, 400], [550, 300], [700, 300], [1280, 400], [1280, 720]], np.int32),
    'roi_lower': np.array([[0, 720], [0, 450], [550, 350], [700, 350], [1280, 450], [1280, 720]], np.int32),
    'roi': np.array([[0, 720], [0, 450], [550, 350], [700, 350], [1280, 450], [1280, 720]], np.int32),
    'canny_t1': 200,
    'canny_t2': 300,
    'hough_threshold': 180,
    'hough_min_line_length': 50,
    'hough_max_line_gap': 25,
    'n_longest_lines': 5,
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

    edges = cv.Canny(img, config['canny_t1'], config['canny_t2'])

    blurred_img = cv.GaussianBlur(edges, (3, 3), 0)
    masked_img = set_roi(blurred_img)

    lines = hough_lines(masked_img)

    if lines is None:
        return edges, None

    lanes = find_lanes(lines)

    img_lines = draw_lines(masked_img.copy(), lanes)

    return img_lines, lanes


def set_roi(img):
    """
    Mask the image, retain specified region of interest.

    :param img: the image to be masked
    :return: masked image
    """

    mask = np.zeros_like(img)
    cv.fillConvexPoly(mask, config['roi'], (255, 255, 255))
    masked = cv.bitwise_and(img, mask)
    return masked


def hough_lines(edges):
    """
    Fit lines to the edges in the image.

    :param edges: the result of Canny's edge detector (blurred)
    :return: an array of lines found by Hough transform
    """

    lines = cv.HoughLinesP(edges, 1, np.pi / 180, config['hough_threshold'],
                           config['hough_min_line_length'], config['hough_max_line_gap'])
    if lines is None:
        return None

    return np.array(list(map(lambda x: x[0], lines)))


# TODO potential problem: the lanes are found at the edges of the road and not the lane the vehicle is in
# TODO improve algorithm - more robust estimation
def find_lanes(lines):
    """
    Find the lanes of the road. Ideally, there should be two distinct lanes (edges of the road).
    First, find the N longest lines and divide them into two bins depending on their slope.
    The average slopes for each bin are the lanes.

    :param lines: the lines found by Hough transform
    :return: an array of lanes found in the image
    """

    lanes = []

    # get lengths of all lines
    line_lengths = map(lambda l: np.sqrt((l[2] - l[0]) ** 2 + (l[3] - l[1]) ** 2), lines)

    # choose N longest lines
    longest_lines = np.array([lines[i] for i, _ in
                             sorted(enumerate(line_lengths), key=lambda p: p[1])[::-1][:config['n_longest_lines']]])

    line_slopes = np.array(list(map(lambda l: (l[3] - l[1]) / (l[2] - l[0]), longest_lines)))
    line_slopes = np.array(list(filter(lambda s: -10 < s < 10, line_slopes)))

    # divide lines into two bins based on their slope
    pos, neg = [], []
    for i, slope in enumerate(line_slopes):
        (pos, neg)[slope < 0].append(slope)

    pos_avg = np.mean(pos)
    neg_avg = np.mean(neg)
    # print(pos_avg, neg_avg)

    lanes = longest_lines
    return lanes


def draw_lines(img, lines):
    """
    Draw lines on the image.

    :param img: the image
    :param lines: the lines (provided as a tuple of point coordinates)
    :return: the image with drawn lines
    """

    for line in lines:
        x1, y1, x2, y2 = line
        cv.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)

    return img
