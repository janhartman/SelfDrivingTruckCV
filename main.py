"""
The main function. Invokes other libraries for image processing and direction finding.
Current performance: 10-15 FPS per loop
"""

import os
import time
import cv2 as cv
import numpy as np

from get_keys import get_pressed_key
from screen_grabber import grab_screen
from image_processor import process_image
from direction_finder import DirectionFinder


def main():

    # use P to pause the program while still running the game
    paused = False

    # learning or driving
    learn = True
    drive = False

    file_name = 'training_data.npy'
    if os.path.isfile(file_name):
        training_data = list(np.load(file_name))
        print(training_data)
    else:
        training_data = []

    # initialize the learner/classifier
    if drive:
        finder = DirectionFinder(training_data)

    # time each loop to measure performance
    loop_time = time.time()

    while True:
        key = get_pressed_key()

        if key == 'P':
            paused = not paused
            time.sleep(1)
            continue

        if not paused:
            image = grab_screen((0, 30, 1280, 720))

            processed_image, lanes = process_image(image)

            cv.imshow('Processed image', processed_image)

            """
            if learning, append a new test case to training data
            X = lane data
            Y = pressed key

            otherwise, determine which keys to press in direction_finder
            """
            if learn and None not in (key, lanes):
                training_data.append([lanes, key])
                if len(training_data) % 100 == 0:
                    print("Gathered", len(training_data), "data samples.")
                    np.save(file_name, np.array(training_data))
            elif drive:
                finder.find_direction(lanes)

        print('Loop time: ', time.time() - loop_time)
        loop_time = time.time()

        if cv.waitKey(25) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break

main()
