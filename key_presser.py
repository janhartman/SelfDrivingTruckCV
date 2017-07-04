"""
Send keypresses to the game based on the command (class) to simulate driving.
"""

import time
from direct_keys import press, release, W, A, S, D

from config import config

default_delta = config['timedelta']


def straight(delta):
    press(W)
    release(A)
    release(D)
    time.sleep(delta)
    release(W)


def left(delta):
    press(W)
    press(A)
    release(D)
    time.sleep(delta)
    release(W)
    release(A)


def right(delta):
    press(W)
    press(D)
    release(A)
    time.sleep(delta)
    release(W)
    release(D)


def brake(delta):
    press(S)
    release(W)
    release(A)
    release(D)
    time.sleep(delta)
    release(S)

directions = {
    'W': straight,
    'A': left,
    'S': brake,
    'D': right
}


def go(cmd, factor=1):
    """
    Used to choose a direction based on the received command.

    :param cmd: The command (can be any of W, A, S, D)
    :param factor: how long to hold key
    """
    directions[cmd](factor * default_delta)
