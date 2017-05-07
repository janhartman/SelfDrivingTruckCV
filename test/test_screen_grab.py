import time
import numpy as np

from PIL import ImageGrab
from screen_grabber import grab_screen

win32 = []
pil = []

for _ in range(1000):
    t1 = time.time()
    image = np.array(ImageGrab.grab(bbox=(0, 0, 1280, 720)))
    t2 = time.time()
    image = grab_screen((0, 0, 1280, 720))
    t3 = time.time()

    pil.append(t2-t1)
    win32.append(t3-t2)

print(np.mean(pil), np.std(pil))
print(np.mean(win32), np.std(win32))
