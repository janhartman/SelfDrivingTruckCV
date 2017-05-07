import time

import cv2
import numpy as np
from PIL import ImageGrab


def draw_lines(img, lines):
    if lines is None:
        return

    for line in lines:
        coords = line[0]
        cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255, 255, 255], 3)


def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


def process_img(original_image):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    processed_img = cv2.GaussianBlur(processed_img, (3, 3), 0)
    vertices = np.array([[10, 720], [10, 500], [300, 300], [700, 300], [1280, 500], [1280, 720]], np.int32)
    processed_img = roi(processed_img, [vertices])

    #                       edges
    lines = cv2.HoughLinesP(processed_img, 1, np.pi / 180, 180, np.array([]), 20, 15)
    draw_lines(processed_img, lines)
    return processed_img


def main():
    # last_time = time.time()
    while True:
        screen = np.array(ImageGrab.grab(bbox=(0, 40, 1280, 720)))
        new_screen = process_img(screen)
        # print('Loop took {} seconds'.format(time.time() - last_time))
        # last_time = time.time()
        cv2.imshow('window', new_screen)
        # cv2.imshow('window2', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


main()
