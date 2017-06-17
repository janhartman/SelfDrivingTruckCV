import time
import numpy as np
from sklearn import svm
from sklearn.utils import shuffle
from direct_keys import press, release, W, A, S, D

delta = 0.1

"""
gas_threshold = 0.5
turn_threshold = 0.5
brake_threshold = 0.5
"""

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


direction = {
    'A': left,
    'W': straight,
    'S': brake,
    'D': right
}


class DirectionFinder:
    def __init__(self, training_data):
        """
        Initialize the direction finder by training the model.
        The model is a support vector machine.
        :param training_data: Training data for the model.
                X: lane data
                Y: pressed key
        """

        self.clf = svm.SVC(decision_function_shape='ovo', cache_size=1000, probability=True)
        arr = np.array(self.balance(training_data))
        self.clf.fit(np.float32(arr.T[0].tolist()), arr.T[1])

    def balance(self, training_data):
        """
        Balance the training data.
        :param training_data:
        :return: balanced data
        """
        training_data = shuffle(training_data)

        w = list(filter(lambda x: x[1] == 'W', training_data))
        a = list(filter(lambda x: x[1] == 'A', training_data))
        s = list(filter(lambda x: x[1] == 'S', training_data))
        d = list(filter(lambda x: x[1] == 'D', training_data))

        l = [w, a, d, ] #s

        min_c = min([len(x) for x in l])

        return w[:min_c] + a[:min_c] + s[:min_c] + d[:min_c]

    def find_direction(self, lanes):
        """
        Predict which direction to take (i.e. which key to press).
        :param lanes: numpy array of lane data
        :return:
        """
        lanes = np.float32(lanes).reshape(1, -1)
        cls = self.clf.predict(lanes)[0]

        direction[cls]()

        """
        probs = self.clf.predict_proba(lanes)[0]

        idx = np.argmin(probs)
        if self.clf.classes_[idx] != cls:
            print("Found a different minimum than the predicted class")
        probs = np.ones(probs.shape) - probs
        print(probs, cls)

        # presuming the keys are sorted A D S W
        if probs[3] > gas_threshold:
            straight()
        elif probs[0] > turn_threshold:
            left()
        elif probs[1] > turn_threshold:
            right()
        elif probs[2] > brake_threshold:
            brake()
        # maybe remove?
        else:
            straight()
        """

