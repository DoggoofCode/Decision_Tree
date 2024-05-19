from math import log2


class Tree:
    def __init__(self, depth_limited=6):
        assert depth_limited
        self.root = None
        self.depth_limit = depth_limited

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if depth == self.depth_limit or len(set(y)) == 1:
            return max(set(y), key=y.count)

        best_feature_index, best_threshold = self._find_best_split(X, y)
        if best_feature_index == -1:
            return max(set(y), key=y.count)

        left_x, left_y, right_x, right_y = self._split_data(X, y, best_feature_index, best_threshold)

        left_subtree = self._build_tree(left_x, left_y, depth + 1)
        right_subtree = self._build_tree(right_x, right_y, depth + 1)

        return best_feature_index, best_threshold, left_subtree, right_subtree

    def _calculate_entropy(self, data):
        self._ = 0
        classes = list(set(data))
        entropy = 0
        for c in classes:
            p = data.count(c) / len(data)
            entropy -= p * log2(p)
        return entropy

    def _split_data(self, X, y, feature_index, threshold):
        self._ = 0
        left_x, left_y, right_x, right_y = [], [], [], []
        for index, value in enumerate(X):
            if value[feature_index] <= threshold:
                left_x.append(value)
                left_y.append(y[index])
            else:
                right_x.append(value)
                right_y.append(y[index])
        return left_x, left_y, right_x, right_y

    def _find_best_split(self, X, y):
        best_entropy = INFINITY
        best_feature_index = -1
        best_threshold = None
        for feature_index in range(len(X[0])):
            for i in range(len(X)):
                threshold = X[i][feature_index]
                left_x, left_y, right_x, right_y = self._split_data(X, y, feature_index, threshold)

                entropy = ((len(left_y) / len(y)) * self._calculate_entropy(left_y)) \
                          + ((len(right_y) / len(y)) * self._calculate_entropy(right_y))

                if entropy < best_entropy:
                    best_entropy = entropy
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def _predict_sample(self, sample, node):
        if isinstance(node, str):
            return node

        feature_index, threshold, left_subtree, right_subtree = node
        if sample[feature_index] <= threshold:
            return self._predict_sample(sample, left_subtree)
        else:
            return self._predict_sample(sample, right_subtree)

    def predict(self, X):
        prediction = []
        for sample in X:
            prediction.append(self._predict_sample(sample, self.root))
        return prediction


INFINITY = float('inf')

X_train = [
    ['Sunny', 'Hot', 'High', 'Weak'],
    ['Sunny', 'Hot', 'High', 'Strong'],
    ['Overcast', 'Hot', 'High', 'Weak'],
    ['Rainy', 'Mild', 'High', 'Weak'],
    ['Rainy', 'Cool', 'Normal', 'Weak'],
    ['Rainy', 'Cool', 'Normal', 'Strong'],
    ['Overcast', 'Cool', 'Normal', 'Strong'],
    ['Sunny', 'Mild', 'High', 'Weak'],
    ['Sunny', 'Cool', 'Normal', 'Weak'],
    ['Rainy', 'Mild', 'Normal', 'Weak'],
    ['Sunny', 'Mild', 'Normal', 'Strong'],
    ['Overcast', 'Mild', 'High', 'Strong'],
    ['Overcast', 'Hot', 'Normal', 'Weak'],
    ['Rainy', 'Mild', 'High', 'Strong']
]
y_train = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']


X_test = [
    ['Sunny', 'Hot', 'High', 'Weak'],
    ['Overcast', 'Hot', 'High', 'Weak'],
    ['Rainy', 'Mild', 'High', 'Strong']
]

model = Tree()
model.fit(X_train, y_train)
print(model.predict(X_test))
