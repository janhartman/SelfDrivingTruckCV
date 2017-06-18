import time
import numpy as np
from sklearn import tree
from sklearn.utils import shuffle
from direct_keys import press, release, W, A, S, D

from config import config

delta = config['timedelta']


def straight():
    press(W)
    release(A)
    release(D)
    time.sleep(delta)
    release(W)


def left():
    press(W)
    press(A)
    release(D)
    time.sleep(delta)
    release(W)
    release(A)


def right():
    press(W)
    press(D)
    release(A)
    time.sleep(delta)
    release(W)
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

        self.data = training_data
        self.clf = tree.DecisionTreeClassifier()
        self.data = self.balance(training_data)
        self.data = np.array(shuffle(self.data))
        self.clf.fit(np.float32(self.data.T[0].tolist()), self.data.T[1])
        self.last = [None for _ in range(1)]

        # print(self.data)

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

        l = [w, a, d, ]  # s

        counts = [len(x) for x in l]
        min_c = min(counts)

        print(counts)

        return w[:min_c] + a[:min_c] + s[:min_c] + d[:min_c]

    def find_direction(self, lanes):
        """
        Predict which direction to take (i.e. which key to press).
        :param lanes: numpy array of lane data
        :return:
        """
        if lanes is None:
            """
            if 'W' in self.last:
                self.last = [None] + self.last[1:]
            else:
                self.last = ['W']
                straight()
            """
            print("w")
            straight()
            return

        lanes = np.float32(lanes).reshape(1, -1)
        cls = self.clf.predict(lanes)[0]
        self.last = [cls]
        print(cls)
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

