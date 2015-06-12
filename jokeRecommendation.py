# -*- coding: utf-8 -*-
"""
Joke Recommendation System
CS189 

@author: owenchen
"""


import scipy.io
import numpy as np
import math 
import csv 
from random import randint
from scipy.special import expit
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import random
import glob
import re
from StringIO import StringIO
from PIL import Image

jokeData= scipy.io.loadmat('joke_data/joke_train.mat')

#### K-Means clustering

imageData= scipy.io.loadmat('mnist_data/images.mat')

#Normalizing data
trainingImages = imageData['images']
trainingImagesFlattened = [trainingImages[:,:,i].flatten()/255.0 for i in range(trainingImages.shape[2])]
cluster_sizes = [5, 10, 20]

def distance(a,b):
    return np.sqrt(np.sum(np.square(a-b)))

def assign_cluster(images, centers):
    clusters = []
    loss = 0
    
    for img in images:
        dist = []
        for i, center in enumerate(centers):
            dist.append((distance(img, center), i))
        mini = min(dist, key=lambda i: i[0])
        clusters.append(mini[1])
        loss += mini[0]**2
    return np.array(clusters), loss
            


def recalc_mean(data, labels, centers):
    result = []
    for i in range(centers):
        result.append(np.mean(data[labels==i], axis=0))
    
    return np.array(result)
    
def k_means(data):
    cluster_iters = []
    
    for size in cluster_sizes:
        #k random centroids
        print("Iteration")
        random = np.random.uniform(0, len(data), size)
        clusters = [data[int(idx)] for idx in random]

        cluster_centers = {}
        for x in range(size):
            cluster_centers[x] = []
        #Repeat until convergence
        cluster_labels, loss= assign_cluster(data, clusters)
        #print ("Loss is " + str(loss))
        #For every cluster, recomupute it's mean
        new_cluster_centers = {}
        for i in range(20):
            new_centers = recalc_mean(np.array(data),np.array(cluster_labels),size)
            cluster_labels, loss = assign_cluster(data, new_centers)
            print ("Loss is " + str(loss))
        cluster_iters.append(new_centers)
        print("End iteration")
    return np.array(cluster_iters)
        
        
cluster_centers = k_means(trainingImagesFlattened)
'''
print (len(cluster_centers[0]))

#5 clusters
for i in enumerate(cluster_centers):
    outimg = np.array(i).reshape(28,28).astype(np.uint8)
    im = Image.fromarray(outimg)
    im.save('5test.png')
    

#10 clusters
for i in enumerate(cluster_centers[1]):
    outimg = np.array(i).reshape(28,28).astype(np.uint8)
    im = Image.fromarray(outimg)
    im.save('10test.png')

#20 clusters
for i in enumerate(cluster_centers[2]):
    outimg = np.array(i).reshape(28,28).astype(np.uint8)
    im = Image.fromarray(outimg)
    im.save('20test.png')
'''


###Laten Factor model with SVD and PCA
jokeData = scipy.io.loadmat('joke_data/joke_train.mat')['train']
validation = np.genfromtxt('joke_data/validation.txt', delimiter = ',')
testData = np.genfromtxt('joke_data/query.txt', dtype='uint8', delimiter=',')
 
def latent_factor_model(d):
    joke_zero = np.nan_to_num(jokeData)
    u, s, v = np.linalg.svd(joke_zero, full_matrices=False)
    S = np.diag(s[:d])
    S = scipy.linalg.sqrtm(S)
    #square_S = np.sqrt(S)
    U = u[:, :d]
    V = v[:d]
    X = np.dot(np.dot(U,S), np.dot(S, V))
    MSE = 0
    for i in range(len(jokeData)):
        for j in range(len(jokeData[0])):
            if not np.isnan(jokeData[i][j]):
                MSE += np.square((X[i][j] - jokeData[i][j]))
                
    prediction = []
    expected = []
    
    for v in validation:
        expected.append(v[2])
        #One indexed 
        if X[v[0]-1][v[1]-1] >= 0:
            prediction.append(1)
        else:
            prediction.append(0)
        
        
    test_prediction = []
    for t in testData:
        if X[t[1]-1][t[2]-1] >= 0:
            test_prediction.append(1)
        else:
            test_prediction.append(0)
            
    count = 0.0
    total = 0.0
    for i in enumerate(expected):
        total += 1
        if expected[i[0]] == prediction[i[0]]:
            count += 1
    return [MSE, count/total, test_prediction]

#2.3.3
def latent_model_with_loss(d, iterations, alpha, eta):
    
    U = np.random.normal(0.0, 1.0, (jokeData.shape[0], d))
    V = np.random.normal(0.0, 1.0, (100, d))
    for iteration in iterations:
        dLdu, dLdv = gradient_loss(U,V, jokeData, alpha)
        U -= eta*dLdu
        V -= eta*dLdv
    X = np.dot(U,V)     
      
    MSE = 0
    for i in range(len(jokeData)):
        for j in range(len(jokeData[0])):
            if not np.isnan(jokeData[i][j]):
                MSE += np.square((X[i][j] - jokeData[i][j])) + alpha*np.sum(np.square(u)) + alpha*np.sum(np.square(v))

#Predict accuracy
    prediction = []
    expected = []
    for v in validation:
        expected.append(v[2])
        #One indexed 
        if X[v[0]-1][v[1]-1] >= 0:
            prediction.append(1)
        else:
            prediction.append(0)
    test_prediction = []
    for t in testData:
        if X[t[1]-1][t[2]-1] >= 0:
            test_prediction.append(1)
        else:
            test_prediction.append(0)
    count = 0.0
    total = 0.0
    for i in enumerate(expected):
        total += 1
        if expected[i[0]] == prediction[i[0]]:
            count += 1
    return [MSE, count/total, test_prediction]      

         
def gradient_loss(U, V, joke_data, lambda_val):
    u_deriv = (2 * (np.dot(U, V.T) - np.nan_to_num(joke_data))).dot(V) + (2 * lambda_val * U) 
    v_deriv = (2 * (np.dot(U, V.T) - np.nan_to_num(joke_data))).T.dot(U) + (2 * lambda_val * V)
    return u_deriv,v_deriv    
    
    

print "d = 2,5,10,20 validation accuracy: "

two = latent_factor_model(2)
five = latent_factor_model(5)
ten = latent_factor_model(10)
twenty = latent_factor_model(20)
print  str(two[1])
print str(five[1])
print str(ten[1])
print str(twenty[1])

print "MSE for d=2,5,10,20"
print  str(two[0])
print str(five[0])
print str(ten[0])
print str(twenty[0])
 
#KAggle ====================================================
results = latent_factor_model(20)
test_predictions = results[2]

with open('kaggle_joke_submission.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Id', 'Category'])
    for i in range(len(test_predictions)):
        writer.writerow([i + 1, test_predictions[i]])
    print 'Done writing to kaggle submission '

    

##KNN and average rating Joke Recommendation

jokeData= scipy.io.loadmat('joke_data/joke_train.mat')['train']
validation = numpy.genfromtxt('joke_data/validation.txt', delimiter = ',') #i,j,s, user index, joke, rating
query = numpy.genfromtxt('joke_data/query.txt', delimiter = ',') #id, i, j


def avg_rating_rec(test_set):
    #2.1 Average rating recommendation
    avg_ratings = np.array([np.nanmean((jokeData[:,i])) for i in range(jokeData.shape[1])])
    positives = np.where(avg_ratings>0.0)[0]
    
    joke_predictions = np.zeros(100)
    for j in positives:
        joke_predictions[j] = 1
    
    accuracy = 0.0
    total_validation = len(test_set)
    
    for x in test_set:
        if (joke_predictions[x[1]-1] == x[2]):
            accuracy += 1.0
    return accuracy/total_validation
    
average_acc = avg_rating_rec(validation)
print ("Prediction accuracy on validation via average joke ratings " + str(int(average_acc*100)) + "%")

def sign(a):
    if a >= 0.0:
        return 1
    elif a < 0.0:
        return 0
    else:
        return 0
    
def preference_recommendation(test_data, k):
    
    new_joke_array = np.nan_to_num(jokeData)
    #For testing
#    new_joke_array = new_joke_array[:1000]
#    print ("Testing on shorted joke array")        
    knn_array = knn(new_joke_array, k)
   
    k_nearest = [10, 100, 1000]
    acc_scores = []
    for k_num in k_nearest:
    #Average the ratings of these neighbors (user x K) array
        predictions = [] 
        for user in range(new_joke_array.shape[0])[:100]:
            user_rating = 0.0
            for neighbor in knn_array[user][:k_num]:
                #This is an array
                user_rating += new_joke_array[neighbor]
            predictions.append(user_rating/k_num)
    
        #Test validation with predictions array
        accuracy = 0.0
        total_validation = len(test_data)
        
        for x in test_data:
            #One indexed
            if(sign(predictions[int(x[0])-1][int(x[1])-1]) == x[2]):
                accuracy += 1.0
        acc_scores.append(accuracy/total_validation)
    return acc_scores
    
    
def distance(a,b):
    return np.sqrt(np.sum(np.square(a-b)))

def knn(joke_array, k=1000):
    distance_array = []
    sorted_neighbors = []
    count = 0 
    
    #1 indexed for jokes
    for user in joke_array[:100]:
        copied_user_array = np.tile(user,(joke_array.shape[0], 1))
        user_dist_array = (np.square(np.subtract(copied_user_array, joke_array)))
        user_dist_summed = [np.sqrt(np.sum(user_dist_array[i])) for i in range(np.array(user_dist_array).shape[0])] 
        sorted_user_indices = numpy.argsort(user_dist_summed)
        distance_array.append(user_dist_array)
        sorted_neighbors.append(sorted_user_indices[1:k])

    return np.array(sorted_neighbors)
   
preference_accuracy = preference_recommendation(validation, k=1000)

print ("Knn accuracy, k = 10, 100, 1000:")
for acc in preference_accuracy:
    print (str(int(acc*100)) + "%")
    


