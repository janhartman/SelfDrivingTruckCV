import cv2 as cv

imgs = []


def create_video():
    out = cv.VideoWriter('output.avi', -1, 15.0, (1281, 721))

    for frame in imgs:
        out.write(frame)

    out.release()
    print('Finished writing video to file')
