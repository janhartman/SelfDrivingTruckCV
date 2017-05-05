from image_processor import *
import cv2 as cv

img1 = cv.imread('screens/Screenshot_1.png')
img2 = cv.imread('screens/Screenshot_2.png')
img3 = cv.imread('screens/Screenshot_3.png')

process_image(img3)
