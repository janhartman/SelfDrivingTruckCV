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

        if len(training_data) == 0:
            return

        self.num_cmds = 0

        self.off_left = False
        self.off_right = False

        # the training data - balance and shuffle
        self.data = self.balance(training_data)
        self.data = np.array(shuffle(self.data))

        # last N commands
        self.last = [None for _ in range(config['n_last_commands'])]

        # the classifier (a decision tree using Gini impurity)
        self.clf = tree.DecisionTreeClassifier(min_samples_split=3, min_samples_leaf=1)
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
        d = list(filter(lambda x: x[1] == 'D', training_data))

        l = [w, a, d]

        counts = [len(x) for x in l]
        min_c = min(counts)

        return w[:min_c] + a[:min_c] + d[:min_c]

    def find_direction(self, lanes):
        """
        Predict which direction to take (i.e. which key to press).
        :param lanes: the lanes
        """

        self.num_cmds += 1

        self.check_off()

        # Start with applying gas.
        if self.num_cmds < 15:
            self.take_direction('W')

        # Brake every n-th turn.
        elif 'S' not in self.last:  # and not (self.off_right or self.off_left):
            if self.num_cmds > 30:
                self.take_direction('S', 2)
            else:
                self.take_direction('S')

        # Check the lanes for validity (if invalid, the default choice is to apply gas).
        elif not self.check_lanes(lanes):  # and not (self.off_right or self.off_left):
            self.take_direction('W')

        # Left lane not found.
        elif lanes[0] == -666 and lanes[1] == -666:
            print('left not found ', end='')

            if self.off_left or lanes[2] > 1 or lanes[3] < -1000:
                self.take_direction('D')
                self.off_left = True
                self.off_right = False
            else:
                self.find_with_clf(lanes)

        # Right lane not found.
        elif lanes[2] == 666 and lanes[3] == 666:
            print('right not found ', end='')

            if self.off_right or lanes[0] < -1 or lanes[1] > 1000:
                self.take_direction('A')
                self.off_right = True
                self.off_left = False
            else:
                self.find_with_clf(lanes)

        # Use the classifier to determine direction.
        else:
            self.find_with_clf(lanes)

    def take_direction(self, direction, factor=1):
        print(direction)
        self.last = [direction] + self.last[:-1]
        go(direction, factor)

    @staticmethod
    def check_lanes(lanes):
        """
        Checks if the lanes are valid.
        The slope/intercept must fall within specified intervals and both lanes must exist.
        :param lanes: the lanes
        :return: the validity of the lanes (boolean)
        """
        # return not (lanes is None or lanes[1] > 1500 or lanes[1] < 0 or lanes[3] < -150)
        return lanes is not None and lanes is not [-666, -666, 666, 666]

    def check_off(self):
        if self.off_right:
            if self.last.count('A') == len(self.last) - 1:
                self.off_right = False
        elif self.off_left:
            if self.last.count('D') == len(self.last) - 1:
                self.off_left = False

    def find_with_clf(self, lanes):
        """
        Find the right direction using the classifier.
        :param lanes: the lanes
        """

        self.off_left = self.off_right = False
        print(lanes)
        lanes = np.array(lanes).reshape(1, -1)
        cls = self.clf.predict(lanes)[0]
        self.take_direction(cls)
