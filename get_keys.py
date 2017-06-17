# Citation: Box Of Hats (https://github.com/Box-Of-Hats )

import win32api as wapi

key_list = ['A', 'W', 'S', 'D', 'P']


def get_pressed_key():
    """
    Check for a pressed key.
    :return: the first detected currently pressed key
    """
    for key in key_list:
        if wapi.GetAsyncKeyState(ord(key)):
            return key
    return None
