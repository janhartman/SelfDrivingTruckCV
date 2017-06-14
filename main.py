"""
The main function. Invokes other libraries for image processing and direction finding.
Current performance: 10-15 FPS per loop
"""

import os
import time
import cv2 as cv
import numpy as np

from get_keys import key_check, map_keys
from screen_grabber import grab_screen
from image_processor import process_image
from direction_finder import DirectionFinder


def main():
    paused = False

    # learning or driving
    learn = False

    file_name = 'training_data.npy'
    if os.path.isfile(file_name):
        print('File exists, loading previous data!')
        training_data = list(np.load(file_name))
    else:
        print('File does not exist, starting fresh!')
        training_data = []

    # initialize the learner/classifier
    if not learn:
        finder = DirectionFinder(training_data)

    loop_time = time.time()

    while True:
        keys = key_check()

        if 'P' in keys:
            paused = not paused
            time.sleep(1)
            continue

        if not paused:
            image = grab_screen((0, 0, 1280, 720))

            processed_image, lanes = process_image(image)

            cv.imshow('Processed image', processed_image)

            if lanes is None:
                continue

            """
            if learning, append a new test case to training data
            X = lane data
            Y = four binary values to indicate which keys were pressed at that time

            otherwise, determine which keys to press in direction_finder
            """
            if learn:
                keys = list(map(int, map_keys(keys)))
                training_data.append([lanes, keys])
                if len(training_data) % 500 == 0:
                    print(len(training_data))
                    np.save(file_name, training_data)
            else:
                finder.find_direction(lanes)

        # print('Loop time: ', time.time() - loop_time)
        # loop_time = time.time()

        if cv.waitKey(25) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break

main()
