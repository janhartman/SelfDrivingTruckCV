"""
The main function. Invokes other libraries for image processing and direction finding.
Current performance: 10-15 FPS per loop
"""

import time
import cv2 as cv

from screen_grabber import grab_screen
from image_processor import process_image


def main():
    # last_time = time.time()

    while True:
        image = grab_screen((0, 0, 1280, 720))

        processed_image, lanes = process_image(image)

        cv.imshow('Processed image', processed_image)

        # print('Loop time: ', time.time() - last_time)
        # last_time = time.time()

        if cv.waitKey(25) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break

main()
