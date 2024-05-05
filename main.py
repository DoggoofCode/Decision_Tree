class Tree:
    def __init__(self, depth_limited=None):
        assert depth_limited
        self.root = None
        self.depth_limit = depth_limited

    def fit(self):
        self.root = 0

    def _build_tree(self, X, y, depth):
        pass

    def _calculate_entropy(self, data):
        pass

    def _split_data(self, X, y, feature_index, threshold):
        pass

    def _find_best_split(self, X, y):
        pass

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
