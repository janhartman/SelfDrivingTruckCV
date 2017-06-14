import time
import numpy as np
from sklearn import svm
from direct_keys import press, release, W, A, S, D


delta = 0.1
gas_threshold = 0.5
turn_threshold = 0.5
brake_threshold = 0.5


def straight():
    press(W)
    release(A)
    release(D)


def left():
    press(W)
    press(A)
    release(D)
    time.sleep(delta)
    release(A)


def right():
    press(W)
    press(D)
    release(A)
    time.sleep(delta)
    release(D)


def brake():
    press(S)
    release(W)
    release(A)
    release(D)
    time.sleep(delta)
    release(S)


class DirectionFinder:
    def __init__(self, training_data):
        """
        Initialize the direction finder by training the model or loading a pre-trained model.
        :param training_data:
        """

        self.clf = svm.SVC(decision_function_shape='ovo', cache_size=1000)
        arr = np.array(training_data)
        self.clf.fit(arr.T[0], arr.T[1])

    def find_direction(self, case):
        probs = self.clf.predict_proba(case)




