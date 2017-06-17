from image_processor import *

img1 = cv.imread('screens/Screenshot_1.png', cv.CV_8UC1)
img2 = cv.imread('screens/Screenshot_2.png', cv.CV_8UC1)
img3 = cv.imread('screens/Screenshot_3.png', cv.CV_8UC1)

for img in [img1, img2, img3]:
    img_lines, lanes = process_image(img)

    if lanes is None:
        print("No lanes found in image")
        exit(0)

    img_lanes = img_lines.copy()

    for lane in lanes:
        x1, y1, x2, y2 = lane
        cv.line(img_lanes, (x1, y1), (x2, y2), (255, 255, 255), 5)

    cv.imshow("img_lanes image", img_lanes)
    cv.waitKey(0)
