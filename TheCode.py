import numpy as np
import pandas as pd
from collections import Counter

class Node:
    def __init__(self, label=None, feature=None, threshold=None):
        self.label = label
        self.feature = feature
        self.threshold = threshold
        self.children = {}
        self.entropy = None
        self.information_gain = None

def entropy(data):
    labels = data['Passed'].values
    label_counts = Counter(labels)
    total_samples = len(labels)
    entropy = -sum((count / total_samples) * np.log2(count / total_samples) for count in label_counts.values())
    return entropy

def information_gain(data, feature):
    total_entropy = entropy(data)
    unique_values = data[feature].unique()
    weighted_entropy = 0

    for value in unique_values:
        subset = data[data[feature] == value]
        subset_entropy = entropy(subset)
        weight = len(subset) / len(data)
        weighted_entropy += weight * subset_entropy

    information_gain = total_entropy - weighted_entropy
    return information_gain

def id3(data, features):
    if len(set(data['Passed'])) == 1:
        return Node(label=data['Passed'].values[0])

    if len(features) == 0:
        label_counts = Counter(data['Passed'])
        majority_label = label_counts.most_common(1)[0][0]
        return Node(label=majority_label)

    best_feature = max(features, key=lambda feature: information_gain(data, feature))
    node = Node(feature=best_feature)

    node.entropy = entropy(data)
    node.information_gain = information_gain(data, best_feature)

    if data[best_feature].dtype == 'O':
        unique_values = data[best_feature].unique()
        for value in unique_values:
            subset = data[data[best_feature] == value]
            if len(subset) == 0:
                label_counts = Counter(data['Passed'])
                majority_label = label_counts.most_common(1)[0][0]
                node.children[value] = Node(label=majority_label)
            else:
                new_features = [f for f in features if f != best_feature]
                node.children[value] = id3(subset, new_features)
    else:
        threshold = data[best_feature].median()
        node.threshold = threshold
        left_subset = data[data[best_feature] <= threshold]
        right_subset = data[data[best_feature] > threshold]

        new_features = [f for f in features if f != best_feature]
        node.children['<= ' + str(threshold)] = id3(left_subset, new_features)
        node.children['> ' + str(threshold)] = id3(right_subset, new_features)

    return node

data = {
    'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'Study Hours': ['2-4 hours', '1-2 hours', '2-4 hours', '5+ hours', '5+ hours',
                    '1-2 hours', '2-4 hours', '2-4 hours', '1-2 hours', '5+ hours',
                    '2-4 hours', '1-2 hours', '1-2 hours', '2-4 hours', '1-2 hours',
                    '2-4 hours', '1-2 hours', '5+ hours', '2-4 hours', '2-4 hours'],
    'Attendance': ['low', 'average', 'high', 'average', 'high', 'average', 'high', 'high', 'average', 'low',
                    'high', 'low', 'high', 'high', 'average', 'high', 'low', 'high', 'high', 'low'],
    'Previous Grade': ['bad', 'good', 'good', 'good', 'good', 'good', 'good', 'bad', 'bad', 'bad',
                       'good', 'good', 'good', 'bad', 'bad', 'bad', 'good', 'good', 'bad', 'bad'],
    'Passed': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes',
               'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes']
}

df = pd.DataFrame(data)
features = ['Study Hours', 'Attendance', 'Previous Grade']
decision_tree = id3(df, features)

def print_tree(node, indent="", is_last_child=True):
    if node.label:
        label_str = "Class: " + node.label
        print(indent + label_str)
    else:
        feature_str = "column: " + node.feature
        print(indent + feature_str)

        child_count = len(node.children)
        for i, (value, child) in enumerate(node.children.items()):
            is_last = i == child_count - 1
            value_str = "Value: " + value
            print(indent + ("└── " if is_last else "├── ") + value_str)
            print_tree(child, indent + ("    " if is_last else "│   "), is_last)

print_tree(decision_tree)
