"""
Used to visualize training data (draw lanes on image and print the class).
"""

import cv2 as cv
import numpy as np
from config import config
from image_processor import draw_lines, make_line_points
from direction_finder import DirectionFinder

data = np.load(config['filename'])
black = np.zeros((config['height'], config['width']))

for l in data:
    lanes = l[0]
    if DirectionFinder.check_lanes(lanes):
        cls = l[1]
        lines = [make_line_points(config['height'], 420, lanes[:2]), make_line_points(config['height'], 420, lanes[2:])]
        img_lines = draw_lines(black.copy(), lines)
        cv.putText(img_lines, cls, (640, 620), cv.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2, cv.LINE_AA)

        print(lanes, cls)
        cv.imshow("lanes", img_lines)
        cv.waitKey(0)
