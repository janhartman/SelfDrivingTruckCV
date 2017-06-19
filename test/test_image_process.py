from image_processor import *

img1 = cv.imread('screens/Screenshot_1.png')
img2 = cv.imread('screens/Screenshot_2.png')
img3 = cv.imread('screens/Screenshot_3.png')

for img in [img1, img2, img3]:
    img_lines, lanes = process_image(img)
    print(lanes)

    if lanes is None:
        print("No lanes found in image")
        continue

    """
    img_lanes = img_lines.copy()
    lanes = [make_line_points(720, 420, lanes[:2]), make_line_points(720, 420, lanes[2:])]

    for lane in lanes:
        x1, y1, x2, y2 = lane
        cv.line(img_lanes, (x1, y1), (x2, y2), (255, 255, 255), 5)
    """
    cv.imshow("img_lanes image", img_lines)
    cv.waitKey(0)
