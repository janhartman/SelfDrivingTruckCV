# TODO potential problem: the lanes are found at the edges of the road and not the lane the vehicle is in
# TODO improve algorithm - more robust estimation
def find_lanes(lines):
    """
    Find the lanes of the road. Ideally, there should be two distinct lanes (edges of the road).
    First, find the N longest lines and divide them into two bins depending on their slope.
    The average slopes for each bin are the lanes.

    :param lines: the lines found by Hough transform
    :return: an array of lanes found in the image
    """

    lanes = []

    # get lengths of all lines
    line_lengths = map(lambda l: np.sqrt((l[2] - l[0]) ** 2 + (l[3] - l[1]) ** 2), lines)

    # choose N longest lines
    longest_lines = np.array([lines[i] for i, _ in
                             sorted(enumerate(line_lengths), key=lambda p: p[1])[::-1][:config['n_longest_lines']]])
    """
    line_slopes = np.array(list(map(lambda l: (l[3] - l[1]) / (l[2] - l[0]), longest_lines)))
    line_slopes = np.array(list(filter(lambda s: -10 < s < 10, line_slopes)))

    # divide lines into two bins based on their slope
    pos, neg = [], []
    for i, slope in enumerate(line_slopes):
        (pos, neg)[slope < 0].append(slope)

    pos_avg = np.mean(pos)
    neg_avg = np.mean(neg)
    # print(pos_avg, neg_avg)
    """
    lanes = longest_lines

    return np.array(lanes)

def transform(img):
    """
    Calculates the perspective (bird's eye) transform of an image.
    :param img: the input image
    :return: the transformed image
    """

    new_w = 480
    new_h = 720
    src_points = config['perspective_roi']
    dst_points = np.float32([[0, new_h], [0, 0], [new_w, 0], [new_w, new_h]])
    m = cv.getPerspectiveTransform(src_points, dst_points)

    """
    vecs = np.float32(list(map(lambda l: [l[2]-l[0], l[3]-l[1]], lines)))
    print(vecs)
    tf_vecs = cv.perspectiveTransform(vecs[None, :, :], m)[0]
    print(tf_vecs)
    new_vecs = np.int32(list(map(lambda l, v: [l[0], l[1], l[0]+v[0], l[1]+v[1]], lines, tf_vecs)))
    print(new_vecs)

    tf_vecs_img = draw_lines(set_roi(img, config['perspective_roi']), new_vecs)
    cv.imshow('vectors', tf_vecs_img)
    cv.waitKey(0)
    """
    tf_img = cv.warpPerspective(img, m, (new_w, new_h))

    # cv.imshow('tf_img', tf_img)
    # cv.waitKey(0)

    return tf_img