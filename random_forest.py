from collections import Counter
import numpy as np

# Decision Tree Node
class DecisionNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain

        # for leaf node
        self.value = value

# Tree building functions
def entropy(y):
    class_labels = np.unique(y)
    entropy = 0
    for cls in class_labels:
        p_cls = len(y[y == cls]) / len(y)
        entropy -= p_cls * np.log2(p_cls)
    return entropy

def split_data(X, y, feature_index, threshold):
    left_indices = np.argwhere(X[:, feature_index] <= threshold).flatten()
    right_indices = np.argwhere(X[:, feature_index] > threshold).flatten()
    return (X[left_indices], y[left_indices]), (X[right_indices], y[right_indices])

def calculate_info_gain(parent, l_child, r_child):
    weight_l = len(l_child[1]) / len(parent[1])
    weight_r = len(r_child[1]) / len(parent[1])
    gain = entropy(parent[1]) - (weight_l * entropy(l_child[1]) + weight_r * entropy(r_child[1]))
    return gain

def best_split(X, y, features_indices):
    best_split = {}
    max_info_gain = -float("inf")

    # loop over all the features
    for feature_index in features_indices:
        feature_values = X[:, feature_index]
        possible_thresholds = np.unique(feature_values)
        # loop over all the feature values present in the data
        for threshold in possible_thresholds:
            # get current split
            (X_left, y_left), (X_right, y_right) = split_data(X, y, feature_index, threshold)
            # check if childs are not null
            if len(X_left) > 0 and len(X_right) > 0:
                current_info_gain = calculate_info_gain((X, y), (X_left, y_left), (X_right, y_right))
                if current_info_gain > max_info_gain:
                    best_split["feature_index"] = feature_index
                    best_split["threshold"] = threshold
                    best_split["left"] = (X_left, y_left)
                    best_split["right"] = (X_right, y_right)
                    best_split["info_gain"] = current_info_gain
                    max_info_gain = current_info_gain

    return best_split

def build_tree(X, y, min_samples_split, max_depth, current_depth=0, features_indices=None):
    num_samples, num_features = np.shape(X)
    # init best split
    best_split_result = {}
    # if dataset is pure
    if len(np.unique(y)) == 1:
        leaf_value = np.unique(y)[0]
        return DecisionNode(value=leaf_value)

    # check if subsample size is reached
    if num_samples >= min_samples_split and current_depth <= max_depth:
        if not features_indices:
            features_indices = range(num_features)

        best_split_result = best_split(X, y, features_indices)
        # check if information gain is positive
        if best_split_result["info_gain"] > 0:
            left_subtree = build_tree(best_split_result["left"][0], best_split_result["left"][1], min_samples_split, max_depth, current_depth + 1, features_indices)
            right_subtree = build_tree(best_split_result["right"][0], best_split_result["right"][1], min_samples_split, max_depth, current_depth + 1, features_indices)
            return DecisionNode(best_split_result["feature_index"], best_split_result["threshold"], left_subtree, right_subtree, best_split_result["info_gain"])

    # compute leaf node
    leaf_value = max(y, key=list(y).count)
    return DecisionNode(value=leaf_value)

# Decision Tree Classifier
class DecisionTreeClassifier:
    def __init__(self, min_samples_split=2, max_depth=None):  # Tambahkan max_depth
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth  # Tambahkan max_depth

    def fit(self, X, y, features_indices=None):
        self.root = build_tree(X, y, self.min_samples_split, self.max_depth, features_indices=features_indices)  # Tambahkan features_indices

    def predict(self, X):
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions

    def make_prediction(self, x, tree):
        if tree.value is not None:
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

# Random Forest Classifier
class RandomForestClassifier:
    def __init__(self, n_trees=100, min_samples_split=2, max_depth=None, n_features=None):  # Tambahkan max_depth
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth  # Tambahkan max_depth
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier(self.min_samples_split, self.max_depth)  # Tambahkan max_depth
            X_sample, y_sample = self.bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample, features_indices=self.get_random_features(X.shape[1]))  # Tambahkan features_indices
            self.trees.append(tree)

    @staticmethod
    def re_fit(old_model, n_addition_trees, X, y):
        for _ in range(n_addition_trees):
            tree = DecisionTreeClassifier(old_model.min_samples_split, old_model.max_depth)  # Tambahkan max_depth
            X_sample, y_sample = old_model.bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample, features_indices=old_model.get_random_features(X.shape[1]))  # Tambahkan features_indices
            old_model.trees.append(tree)

    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        if self.n_features is not None:
            selected_features = np.random.choice(X.shape[1], size=self.n_features, replace=False)
            return X[indices][:, selected_features], y[indices]
        else:
            return X[indices], y[indices]

    def predict(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_predictions = np.swapaxes(tree_predictions, 0, 1)
        y_pred = [self.most_common_label(tree_prediction) for tree_prediction in tree_predictions]
        return np.array(y_pred)

    def most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def get_random_features(self, n_features):
        if self.n_features is not None:
            return np.random.choice(n_features, size=self.n_features, replace=False)
        else:
            return None
