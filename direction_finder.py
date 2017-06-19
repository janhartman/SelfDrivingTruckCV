import numpy as np
from sklearn import tree
from sklearn.utils import shuffle

from config import config
from key_presser import go


class DirectionFinder:
    def __init__(self, training_data):
        """
        Initialize the direction finder by training the model.
        The model is a decision tree
        :param training_data: Training data for the model.
                X: lane data
                Y: pressed key
        """

        # the training data
        self.data = self.balance(training_data)
        self.data = np.array(shuffle(self.data))

        # last N commands
        self.last = [None for _ in range(config['n_last'])]

        # the classifier (a decision tree using Gini impurity)
        self.clf = tree.DecisionTreeClassifier(min_samples_split=4, min_samples_leaf=1)
        self.clf.fit(np.float32(self.data.T[0].tolist()), self.data.T[1])

    def balance(self, training_data):
        """
        Balance the training data to have an equal distribution of classes.
        :param training_data: array [X | Y] where X are cases and Y classes
        :return: balanced data
        """
        training_data = shuffle(training_data)

        training_data = np.array(list(filter(lambda x: self.check_lanes(x[0]), training_data)))

        print(training_data)

        w = list(filter(lambda x: x[1] == 'W', training_data))
        a = list(filter(lambda x: x[1] == 'A', training_data))
        s = list(filter(lambda x: x[1] == 'S', training_data))
        d = list(filter(lambda x: x[1] == 'D', training_data))

        l = [w, a, d, ]  # s

        counts = [len(x) for x in l]
        min_c = min(counts)

        return w[:min_c] + a[:min_c] + s[:min_c] + d[:min_c]

    def find_direction(self, lanes):
        """
        Predict which direction to take (i.e. which key to press).
        :param lanes: numpy array of lane data
        """

        # Brake every n-th turn.
        if 'S' not in self.last:
            print("s", self.last)
            self.last = ['S'] + self.last[:-1]
            go('S')
            return

        # Check the lanes for validity (if invalid, apply gas).
        if not self.check_lanes(lanes):
            print("w")
            self.last = ['W'] + self.last[:-1]
            go('W')

        # Use the classifier to determine direction.
        else:
            self.find_with_clf(lanes)

    def check_lanes(self, lanes):
        """
        Checks if the lanes are valid.
        The slope/intercept must fall within specified intervals and both lanes must exist. TODO change for both?
        :param lanes: the found lanes
        :return: the validity of the lanes.
        """
        return not (lanes is None or lanes[1] > 1500 or lanes[1] < 0 or lanes[3] < -150)

    def find_with_clf(self, lanes):

        lanes = np.float32(lanes).reshape(1, -1)
        cls = self.clf.predict(lanes)[0]
        print(cls)
        self.last = [cls] + self.last[:-1]
        go(cls)

        """
        # Use probabilities
        probs = self.clf.predict_proba(lanes)[0]

        idx = np.argmax(probs)
        if self.clf.classes_[idx] != cls:
            print("Found a different max than the predicted class")

        print(probs, cls)

        # presuming the keys are sorted A D S W
        if probs[3] > config['gas_threshold']:
            straight()
        elif probs[0] > config['turn_threshold']:
            left()
        elif probs[1] > config['turn_threshold']:
            right()
        elif probs[2] > config['brake_threshold']:
            brake()
        # maybe remove?
        else:
            straight()
        """
