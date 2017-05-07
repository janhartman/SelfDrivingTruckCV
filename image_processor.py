import cv2 as cv
import numpy as np

config = {
    'bbox': (0, 0, 1280, 720),
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


# Process the current screenshot of the game and return the img_lines image.
def process_image(img):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # find edges
    edges = cv.Canny(gray_img, config['canny_t1'], config['canny_t2'])

    blurred_img = cv.GaussianBlur(edges, (3, 3), 0)

    # use only region of interest
    masked_img = set_roi(blurred_img)

    # find lines
    lines = hough_lines(masked_img)

    if lines is None:
        return edges, None

    # find lanes
    lanes = find_lanes(lines)

    img_lines = draw_lines(masked_img.copy(), lanes)

    return img_lines, lanes


# Mask the image to retain only a specified region of interest.
def set_roi(img):
    mask = np.zeros_like(img)
    cv.fillConvexPoly(mask, config['roi'], (255, 255, 255))
    masked = cv.bitwise_and(img, mask)
    return masked


# Fit lines to image.
def hough_lines(edges):
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, config['hough_threshold'],
                           config['hough_min_line_length'], config['hough_max_line_gap'])
    if lines is None:
        return None

    return np.array(list(map(lambda x: x[0], lines)))


# Find the lanes. There should only be two distinct lanes.
# In case of failure, return an empty list
# TODO potential problem: the lanes are found at the edges of the road and not the lane the vehicle is in
def find_lanes(lines):
    lanes = []

    # get lengths of all lines
    line_lengths = map(lambda l: np.sqrt((l[2] - l[0]) ** 2 + (l[3] - l[1]) ** 2), lines)

    # choose N longest lines
    longest_lines = np.array([lines[i] for i, _ in
                             sorted(enumerate(line_lengths), key=lambda p: p[1])[::-1][:config['n_longest_lines']]])

    line_slopes = np.array(list(map(lambda l: (l[3] - l[1]) / (l[2] - l[0]), longest_lines)))
    line_slopes = np.array(list(filter(lambda s: -1. < s < 1., line_slopes)))

    # divide lines into two bins based on their slope
    pos, neg = [], []
    for i, slope in enumerate(line_slopes):
        (pos, neg)[slope < 0].append(slope)

    pos_avg = np.mean(pos)
    neg_avg = np.mean(neg)
    # print(pos_avg, neg_avg)

    lanes = longest_lines
    return lanes


# Draw lines on image.
def draw_lines(img, lines):
    for line in lines:
        x1, y1, x2, y2 = line
        cv.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)

    return img
