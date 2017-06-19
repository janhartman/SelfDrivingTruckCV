"""
Used to issue directions by pressing keys.
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
    'A': left,
    'W': straight,
    'S': brake,
    'D': right
}


def go(cls):
    directions[cls]()
