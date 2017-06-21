"""
The configuration of the program.
"""
config = {
    # image processing
    'roi': [[50, 720], [50, 400], [550, 350], [730, 350], [1210, 400], [1210, 720]],
    'canny_t1': 100,
    'canny_t2': 250,
    'blur_sigma': 3,
    'hough_threshold': 10,
    'hough_min_line_length': 50,
    'hough_max_line_gap': 30,
    'n_longest_lines': 10,

    # window size, general pipeline
    'height': 720,
    'width': 1280,
    'bbox': (0, 30, 1280, 750),
    'filename': 'training_data.npy',
    'keylist': ['A', 'W', 'S', 'D', 'P'],

    # direction finding
    'n_last_commands': 3,
    'timedelta': 0.1,


}
