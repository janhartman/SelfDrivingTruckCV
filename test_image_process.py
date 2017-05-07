from image_processor import *

img1 = cv.imread('screens/Screenshot_1.png')
img3 = cv.imread('screens/Screenshot_3.png')

img = img3

img_lines, lanes = process_image(img)

# cv.imshow("img_lines image", img_lines)
# cv.waitKey(0)

img_lanes = img.copy()
img_lanes = set_roi(img_lanes)

for lane in lanes:
    x1, y1, x2, y2 = lane
    cv.line(img_lanes, (x1, y1), (x2, y2), (255, 255, 255), 5)


cv.imshow("img_lanes image", img_lanes)
cv.waitKey(0)
