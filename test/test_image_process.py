from image_processor import *

img1 = cv.imread('screens/Screenshot_1.jpg')
img2 = cv.imread('screens/Screenshot_2.jpg')
img3 = cv.imread('screens/Screenshot_3.jpg')
img4 = cv.imread('screens/Screenshot_4.jpg')

for img in [img1, img2, img3, img4]:
    img_lines, lanes = process_image(img)
    print(lanes)

    if lanes is None:
        print("No lanes found in image")
        continue

    cv.imshow("img_lanes image", img_lines)
    cv.waitKey(0)
