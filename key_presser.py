"""
Send keypresses to the game based on the command (class) to simulate driving.
"""

import time
from direct_keys import press, release, W, A, S, D

from config import config

delta = config['timedelta']


def straight():
    press(W)
    release(A)
    release(D)
    time.sleep(delta)
    release(W)


def left():
    press(W)
    press(A)
    release(D)
    time.sleep(delta)
    release(W)
    release(A)


def right():
    press(W)
    press(D)
    release(A)
    time.sleep(delta)
    release(W)
    release(D)


def brake():
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


def go(cmd):
    """
    Used to choose a direction based on the received command.

    :param cmd: The command (can be any of W, A, S, D)
    """
    directions[cmd]()
