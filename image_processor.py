import cv2 as cv
import numpy as np

config = {
    'bbox': (0, 0, 1280, 720),
    'roi': np.array([[0, 720], [0, 400], [550, 300], [700, 300], [1280, 400], [1280, 720]], np.int32),
    'canny_t1': 200,
    'canny_t2': 300,
    'hough_threshold': 180,
    'hough_min_line_length': 20,
    'hough_max_line_gap': 25,
}


# Process the current screenshot of the game and return the processed image.
def process_image(img):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # find edges
    edges = cv.Canny(gray_img, config['canny_t1'], config['canny_t2'])

    blurred_img = cv.GaussianBlur(edges, (3, 3), 0)

    # use only region of interest
    masked_img = set_roi(blurred_img)

    # find lines
    lines, img_lines = hough_lines(masked_img)

    # find lanes
    lanes = find_lanes(lines)

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
        return None, edges

    for line in lines:
        coords = line[0]
        cv.line(edges, (coords[0], coords[1]), (coords[2], coords[3]), (255, 255, 255), 3)

    return lines, edges


# Find the lanes.
def find_lanes(lines):


    return []
