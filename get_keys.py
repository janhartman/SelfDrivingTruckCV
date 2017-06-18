"""
Checks which keys are pressed, then returns the first detected pressed key from a specific list of keys.
"""

import win32api as wapi
from config import config

key_list = config['keylist']


def get_pressed_key():
    """
    Check for a pressed key.
    :return: the first detected currently pressed key
    """
    for key in key_list:
        if wapi.GetAsyncKeyState(ord(key)):
            return key
    return None
