"""
The main function. Invokes other libraries for image processing and direction finding.
Current performance: 10-15 FPS while learning, 5-10 FPS while driving.

The program can be used in either learning or driving mode (it is possible to disable both to observe the processing).
The configuration is found in config.py

Inspired by: https://github.com/Sentdex/pygta5
"""

import os
import time

import cv2 as cv
import numpy as np

from config import config
import video_writer as vw
from direction_finder import DirectionFinder
from get_keys import get_pressed_key
from image_processor import process_image
from screen_grabber import grab_screen


def main():

    # use P to pause the program while still running the game
    paused = False

    # visualize frames
    visualize = True

    # learning mode / driving mode (or none of those)
    learn = False
    drive = True

    # produce video of captured screens with drawn lanes
    video = False

    # load training data (learning: append new data, driving: fit model)
    file_name = config['filename']
    training_data = list(np.load(file_name)) if os.path.isfile(file_name) else []

    # initialize the learner/classifier
    finder = DirectionFinder(training_data)

    # time each loop to measure performance
    loop_time = time.time()
    loop_times = []

    while True:
        key = get_pressed_key()

        if key == 'P':
            paused = not paused
            time.sleep(1)
            continue

        if not paused:
            image = grab_screen(config['bbox'])

            processed_image, lanes, lanes_orig_image = process_image(image)

            if visualize:
                cv.imshow('Processed image', processed_image)

            if video:
                vw.imgs.append(lanes_orig_image)

            """
            if learning, append a new test case to training data
            X = lane data
            Y = pressed key

            otherwise, determine which keys to press in direction_finder
            """
            if learn and None not in (key, lanes):
                training_data.append([lanes, key])

                if len(training_data) % 100 == 0:
                    print('Gathered', len(training_data), 'data samples.')
                    np.save(file_name, np.array(training_data))

            if drive:
                finder.find_direction(lanes)

        last_loop_time = time.time() - loop_time
        loop_times.append(last_loop_time)
        loop_time = time.time()
        # print('Loop time: ', last_loop_time)

        if cv.waitKey(25) & 0xFF == ord('q'):
            cv.destroyAllWindows()

            mean_loop_time = np.mean(loop_times)
            print()
            print('Mean loop time:', mean_loop_time)
            print('FPS:', 1 / mean_loop_time)

            if video:
                print(len(vw.imgs), 'frames taken')
                vw.create_video()
            break


if __name__ == '__main__':
    main()
