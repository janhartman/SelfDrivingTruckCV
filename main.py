import cv2 as cv
import numpy as np

from PIL import ImageGrab
from image_processor import process_image


def main():
    while True:
        screen = np.array(ImageGrab.grab(bbox=(0, 0, 1280, 720)))
        new_screen, lanes = process_image(screen)
        cv.imshow('window', new_screen)

        if cv.waitKey(25) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break

main()
