# Citation: Box Of Hats (https://github.com/Box-Of-Hats )

import win32api as wapi

key_list = ["A", "W", "S", "D", "P"]


def key_check():
    keys = set()
    for key in key_list:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.add(key)
    return list(keys)


def map_keys(keys):
    return ["W" in keys, "A" in keys, "S" in keys, "D" in keys]
