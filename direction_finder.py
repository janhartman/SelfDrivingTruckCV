import numpy as np
from sklearn import tree
from sklearn.utils import shuffle

from config import config
from key_presser import go


class DirectionFinder:
    def __init__(self, training_data):
        """
        Initialize the direction finder by training the model.
        The model is a decision tree using Gini impurity as the criterion.
        :param training_data: Training data for the model.
                X: lane data (slope and intercept for each of the two lanes)
                Y: pressed key
        """

        self.cmds = 0

        # the training data - balance and shuffle
        self.data = self.balance(training_data)
        self.data = np.array(shuffle(self.data))

        # last N commands
        self.last = [None for _ in range(config['n_last_commands'])]

        # the classifier (a decision tree using Gini impurity)
        self.clf = tree.DecisionTreeClassifier(min_samples_split=3, min_samples_leaf=1, max_depth=5)
        self.clf.fit(np.float32(self.data.T[0].tolist()), self.data.T[1])

    def balance(self, training_data):
        """
        Balance the training data to have an equal distribution of classes.
        :param training_data: array [X | Y] where X are cases and Y classes
        :return: balanced data
        """

        training_data = shuffle(training_data)

        training_data = list(filter(lambda x: self.check_lanes(x[0]), training_data))

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
        :param lanes: the lanes
        """

        self.cmds += 1

        # start with applying gas
        if self.cmds < 10:
            self.last = ['W'] + self.last[:-1]
            go('W')

        # Brake every n-th turn.
        elif 'S' not in self.last:
            print('S')
            self.last = ['S'] + self.last[:-1]
            go('S')

        # Check the lanes for validity (if invalid, the default choice is to apply gas).
        elif not self.check_lanes(lanes):
            print('w')
            self.last = ['W'] + self.last[:-1]
            go('W')

        # Use the classifier to determine direction.
        else:
            self.find_with_clf(lanes)

    @staticmethod
    def check_lanes(lanes):
        """
        Checks if the lanes are valid.
        The slope/intercept must fall within specified intervals and both lanes must exist.
        :param lanes: the lanes
        :return: the validity of the lanes
        """

        return not (lanes is None or lanes[1] > 1500 or lanes[1] < 0 or lanes[3] < -150)

    def find_with_clf(self, lanes):
        """
        Find the right direction using the classifier.
        :param lanes:
        :return:
        """

        lanes = np.array(lanes).reshape(1, -1)
        cls = self.clf.predict(lanes)[0]
        print(cls)
        self.last = [cls] + self.last[:-1]
        go(cls)
