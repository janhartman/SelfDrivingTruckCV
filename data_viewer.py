"""
Used to visualize training data (draw lanes on image and print the class).
"""

import cv2 as cv
import numpy as np
from config import config
from image_processor import draw_lines, make_line_points

data = np.load('training_data.npy')
black = np.zeros((config['height'], config['weight']))

for l in data:
    lanes = l[0]
    cls = l[1]
    lines = [make_line_points(config['height'], 420, lanes[:2]), make_line_points(config['height'], 420, lanes[2:])]
    img_lines = draw_lines(black.copy(), lines)
    print(lanes, cls)
    cv.imshow("lanes", img_lines)
    cv.waitKey(0)
