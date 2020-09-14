from math import log
import numpy as np
from time import time
from collections import Counter
import sys

class Tree:
    leaf = True	# boolean if it is a final leaf of a tree or not
    prediction = None # what is the prediction (if leaf)
    feature = None # which feature to split on?
    threshold = None # what threshold to split on?
    left = None # left subtree
    right = None # right subtree
    
    def __init__(self):
        self.leaf = True
        self.prediction = None
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None

class Data:
    
    features = [] # list of lists (size: number_of_examples x number_of_features)
    labels = [] # list of strings (lenght: number_of_examples)
    
    def __init__(self):
        self.features = []
        self.lebels = []
    
###################################################################

def read_data(txt_path):
    # TODO: function that will read the .txt file and store it in the data structure
    # use the Data class defined above to store the information

    data = Data()
    data.features = []
    data.labels = []
    f = open(txt_path,'r')
    out = f.readlines()
    for line in out:
        x = line.split(",")
        data.features.append([float(i) for i in x[0:4]])
        if x[4] == 'Iris-setosa\n':
            data.labels.append('Iris-setosa')
        elif x[4] == 'Iris-versicolor\n':
            data.labels.append('Iris-versicolor')
        else:
            data.labels.append('Iris-virginica')
    f.close()
    return data

def predict(tree, point):
    # TODO: function that should return a prediction for the specific point (predicted label)
    # base case
    if(tree.leaf):
        return tree.prediction
    elif point[tree.feature] <= tree.threshold and tree.left:
        return predict(tree.left, point)
    else:
        return predict(tree.right, point)

def split_data(data, feature, threshold):
    # TODO: function that given specific feature and the threshold will divide the data into two parts
    left = Data()
    left.features = []
    left.labels = []
    right = Data()
    right.features = []
    right.labels = []
    # i am assuming feature uses index 
    for i in range(len(data.labels)):
        if data.features[i][feature] <= threshold:
            left.features.append(data.features[i])
            left.labels.append(data.labels[i])
        else:
            right.features.append(data.features[i])
            right.labels.append(data.labels[i])
    return (left, right)

def get_entropy(data):
    # TODO: calculate entropy given data
    setosa=0.0
    versicolor=0.0
    virginica=0.0
    for i in data.labels:
        if i == 'Iris-setosa':
            setosa=setosa+1
        if i == 'Iris-versicolor':
            versicolor=versicolor+1
        if i == 'Iris-virginica':
            virginica=virginica+1
    total=setosa+versicolor+virginica
#     print(total)
    prob_setosa=setosa/total
    prob_versicolor=versicolor/total
    prob_virginica=virginica/total
    
    entropy = 0
    if setosa != 0:
        # print("here ", total)
        entropy = entropy - 1*prob_setosa*(log(prob_setosa)/log(2))
    if versicolor != 0:
        entropy = entropy - 1*prob_versicolor*(log(prob_versicolor)/log(2))
    if virginica != 0:
        -1*prob_virginica*(log(prob_virginica)/log(2))
    return entropy

def find_best_threshold(data, feature):
    # TODO: iterate through data (along a single feature) to find best threshold (for a specified feature)
    entropy = get_entropy(data)
    best_threshold = None
    best_gain = 0
    
    for i in range(len(data.labels)):
        # to split in between two features in same class does not make sense
        # but if not applicable delete the if condition
        if (i != 0) and (data.labels[i] == data.labels[i-1]):
            continue
        left, right = split_data(data, feature, data.features[i][feature]) # data.features[i] means a specific point
        if left.features and right.features:
            wrighted_entropy = get_entropy(left)*len(left.features)/len(data.features) + get_entropy(right)*len(right.features)/len(data.features)
            gain = entropy - wrighted_entropy
            if gain > best_gain:
                best_gain = gain
                best_threshold = data.features[i][feature]
    return best_gain, best_threshold

def find_best_split(data):
    # TODO: iterate through data along all features to find the best possible split overall
    best_gain_0, best_threshold_0 = find_best_threshold(data, 0)
    best_gain_1, best_threshold_1 = find_best_threshold(data, 1)
    best_gain_2, best_threshold_2 = find_best_threshold(data, 2)
    best_gain_3, best_threshold_3 = find_best_threshold(data, 3)
    
    best_gain = best_gain_0
    best_feature = 0
    best_threshold = best_threshold_0
    
    if(best_gain_1 > best_gain):
        best_gain = best_gain_1
        best_feature = 1
        best_threshold = best_threshold_1
    if(best_gain_2 > best_gain):
        best_gain = best_gain_2
        best_feature = 2
        best_threshold = best_threshold_2
    if(best_gain_3 > best_gain):
        best_gain = best_gain_3
        best_feature = 3
        best_threshold = best_threshold_3
    
    return best_feature, best_threshold


def c45(data):
    # TODO: Construct a decision tree with the data and return it.
    tree = c45_helper(data, [], len(data.features[0]))
    return tree

def c45_helper(data, used_features_list, num_features): 
    node = Tree() # node is Tree Class
    if len(used_features_list) == num_features: # if no other features
        # if len(set(data.labels)) != 1:
            # print("warning: not pure. reached:", num_features)
            # print(data.labels)
        node.leaf = True
        node.prediction = max(data.labels,key=data.labels.count)
        return node
    elif len(set(data.labels)) == 1: # if the node is pure
        node.leaf = True
        node.prediction = max(data.labels,key=data.labels.count)
        return node
    
    best_feature, best_threshold = find_best_split(data)
    node.leaf = False
    node.prediction = None
    node.feature = best_feature # which feature to split on?
    
    used_features_list.append(best_feature) # append feature to used list
    
    node.threshold = best_threshold # what threshold to split on?
    
    left_data, right_data = split_data(data, best_feature, best_threshold)
    
    node.left = c45_helper(left_data, used_features_list, num_features) # left subtree
    node.right = c45_helper(right_data, used_features_list, num_features) # right subtree
    
    
    return node
def test(data, tree):
    # TODO: given data and a constructed tree - return a list of strings (predicted label for every example in the data)
    predictions = []
    for i in range(len(data.features)):
        predictions.append(predict(tree, data.features[i]))
#         if predict(tree, data.features[i]) == 
        
    return predictions
###################################################################

if __name__ == '__main__':
    data_train = read_data("train_data.txt")
    data_test = read_data("test_data.txt")
    tree = c45(data_train)

    cnt = 0
    preds = test(data_test, tree)
    for i in range(len(data_test.labels)):
        if (data_test.labels[i] == preds[i]):
            cnt = cnt + 1
    print("acc = ", 1.0*cnt/len(data_test.labels))