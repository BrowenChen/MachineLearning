# -*- coding: utf-8 -*-
"""
Decision Tree and Random Forest Spam Classification

Kaggle Competition, 81% accuracy spam classification 
@author: owenchen
"""
import scipy.io
import numpy as np
import math 
import csv 

spamData = scipy.io.loadmat('spam-dataset/spam_data.mat')
train_labels = spamData['training_labels'][0]
training_data = spamData['training_data']
test_data = spamData['test_data']


class DecisionTree:
    def __init__(self, name, depth=100, impurity=None, segmentor=None):
        self.name = name
        self.depth = depth
        self.impurity = impurity
        self.segmentor = segmentor
        self.root = None
        
    def train(self, training_data, train_labels, depth): #S is training set 
        #print ("Depth is " + str(depth))
            
        if all(x == train_labels[0] for x in train_labels):
            #print ("All labels are the same - Leaf")
            return leafNode(train_labels[0])            
        elif depth == 0:
            #print ("Max depth reached - Leaf ")
            most_common = np.bincount(train_labels).argmax()
            return leafNode(most_common)
        else:
            f,t = segmentor(training_data, train_labels, impurity)
            #i is the feature to be split on, t is the threshold, split the dataset based on t
            root = (f,t)
            #print (root)
            #Split training data. 
            left_training, left_labels, right_training,right_labels = [], [], [], []

            for i in range(len(training_data)):
                datum = training_data[i]
                if datum[f] < t:
                    left_training.append(datum)
                    left_labels.append(train_labels[i])
                elif datum[f] >= t:
                    right_training.append(datum)
                    right_labels.append(train_labels[i])                    

            if len(left_labels) == 0:
                #print ("LEFT IS ZERO making Leaf")
                #print (np.sum(np.array(left_labels)))
                return leafNode(np.bincount(right_labels).argmax())
                
            if len(right_labels) == 0:
                #print ("RIGHT IS ZERO making Leaf")
                #print (left_labels)
                #print(np.bincount(left_labels).argmax())
                return leafNode(np.bincount(left_labels).argmax())
                
            left_tree = self.train(left_training, left_labels, depth-1)
            right_tree = self.train(right_training, right_labels, depth-1)
            
            #print ("creating a new tree..")
            d_tree = Node(root, left_tree, right_tree)
            self.root = d_tree
            return d_tree
                    
                
    def predict(self, test_data):
        #Spam or ham?
        #print("Making prediction")   
        predictions = []
        for datum in test_data:
            node = self.root
            while isinstance(node, leafNode) == False:
                node = node.left if datum[node.split[0]] < node.split[1] else node.right
            if isinstance(node, leafNode):
                #print ("Made a prediction " + str(node.label))
                predictions.append(node.label)
                
        return predictions
                    
                    
        
class Node:
    """
    Each parent node has children
    """
    def __init__(self, split_rule, left, right):
        self.split = split_rule
        self.left = left
        self.right = right
        
class leafNode:
    
    def __init__(self, label):
        #print ("Leaf nodee!!!")
        self.label = label
    
def impurity(left_label_hist, right_label_hist):
    #Takes the results of two splits calculates a scalar impurity 

    p_left = float(np.sum(left_label_hist) / float(len(left_label_hist)))
    p_left_0 = 1 - p_left
    p_right = float(np.sum(right_label_hist) / float(len(right_label_hist)))
    p_right_0 = 1 - p_right
    
    #Calculate average entropy 
    left_e = 0.0 if (p_left == 0.0 or p_left == 1.0) else -p_left * math.log(p_left,2) - p_left_0*math.log(p_left_0,2)
    right_e = 0.0 if (p_right == 0.0 or p_right == 1.0) else -p_right * math.log(p_right,2) - p_right_0*math.log(p_right_0,2) 
    #print ("Calculating Entropy " + str(left_e) + " " + str(right_e))
        
    return (.5 * (left_e + right_e))
    
def segmentor(data, labels, impurity):
    #Find the best split rule for Node 
    #Threshold values mean of features    
    threshold = [np.mean(training_data[:,i]) for i in range(len(training_data[0]))]
    impurities= [] #Impurities for 32 features 
    for feature in range(32):
        hist_l = []
        hist_r = []
        f_thres = threshold[feature]
        
        for i in range(len(data)):
            datum = data[i]
            hist_l.append(labels[i]) if datum[feature] < f_thres else hist_r.append(labels[i])
        impurities.append(impurity(hist_l, hist_r))
    #print (impurities)
    impurities = np.array(impurities)
    min_feature = -1    
    min_so_far = 10.0
    for x in range(len(impurities)):
        if impurities[x] < min_so_far:
            min_feature = x
            min_so_far = impurities[x]
    return (min_feature, threshold[min_feature])

#Random Forest Class
class RandomForest:
    def __init__(self, num_trees, subset_size, depth, rho):
        self.num_trees = num_trees
        self.subset_size = subset_size
        self.depth = depth        
        self.rho = rho
        self.trees = []
        
    def run(self, test_data):
        #Each forest uses a rnadom subset of features 
        ensembles = []
        
        for i in range(self.num_trees):
            tree_idx = np.arange(len(test_data))
            np.random.shuffle(tree_idx)
            t_data = []
            training_subset, label_subset = [], []

            for y in range(self.subset_size):
                t_data.append(test_data[tree_idx[y]])
                training_subset.append(training_data[y])
                label_subset.append(train_labels[y])
                
            classifier = DecisionTree("D-Tree", impurity, segmentor)
            root = classifier.train(training_subset, label_subset, self.depth)
            prediction = classifier.predict(test_data)

            self.trees.append(root)
            ensembles.append(prediction)
            
        ensembles = np.array(ensembles)
        final_prediction = []
        for i in range(len(test_data)):
            final_prediction.append(np.bincount(ensembles[:,i]).argmax())
        return final_prediction
            
label_keys = np.arange(len(train_labels))
train_keys = np.arange(len(training_data))
np.random.shuffle(label_keys)
np.random.shuffle(train_keys)

validation_labels = [train_labels[key] for key in label_keys[:1000]]
validation_data = [training_data[key] for key in train_keys[:1000]]

train_set = [training_data[key] for key in train_keys[1000:]]
train_label_set = [train_labels[key] for key in label_keys[1000:]]
            
#Predicting Test Labels 
print("Running D-Tree")
classifier = DecisionTree("D-Tree", impurity, segmentor)
root = classifier.train(train_set, train_label_set, 150)

print ("D-Tree root split")
print (root.split)
print ("D-Tree Predicted Labels")

print (" ")
predictions = classifier.predict(test_data)
#print (np.array(predictions))
print(predictions)

#Random Forests
print(" ")
print("Running random Forest Predictions")
forest = RandomForest(100, 3000, 20, 15)
forest_predictions = forest.run(test_data)


print(forest_predictions)
#print(np.arrya(forest_predictions))
'''
print("Validation Data")
val_dtree_predictions = classifier.predict(validation_data)
val_forest_predictions = forest.run(validation_data)

print(val_dtree_predictions)
print (" ")
print(val_forest_predictions)
'''
print("Writing to kaggle spam submission")

with open('kaggle_spam_submission.csv', 'wb') as csvfile:
	predictionWriter = csv.writer(csvfile, delimiter=",")
	predictionWriter.writerow(['Id','Category'])
	predictionId = 1
	for label in predictions:
         predictionWriter.writerow([predictionId, label])
         predictionId+=1


print("Writing to forest spam submission")

with open('forest_spam.csv', 'wb') as csvfile:
	predictionWriter = csv.writer(csvfile, delimiter=",")
	predictionWriter.writerow(['Id','Category'])
	predictionId = 1
	for label in forest_predictions:
         predictionWriter.writerow([predictionId, label])
         predictionId+=1




