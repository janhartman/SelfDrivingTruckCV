import numpy as np

"""
The configuration.
"""
config = {
    'roi': np.int32([[50, 720], [50, 450], [550, 350], [730, 350], [1210, 450], [1210, 720]]),
    'canny_t1': 100,
    'canny_t2': 250,
    'blur_sigma': 3,
    'hough_threshold': 10,
    'hough_min_line_length': 50,
    'hough_max_line_gap': 30,
    'n_longest_lines': 10,

    'bbox': (0, 30, 1280, 720),
    'filename': 'training_data.npy',
    'keylist': ['A', 'W', 'S', 'D', 'P'],
    'timedelta': 0.1,


}