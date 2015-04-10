# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:27:54 2015
@author: owenchen

Stochastic/Batch Gradient Methods of Classifying spam or ham

"""
from scipy import io
import numpy as np
import random as rand
import matplotlib.pyplot as plt
#Problem 3

#57 Features
dataSet = io.loadmat('spam.mat')

ytrain = dataSet['ytrain']

#data points x features
Xtest  = dataSet['Xtest']
train  = dataSet['Xtrain']
'''
train_word_match = Xtrain[:,:48]
train_char_match = Xtrain[:,49:55]
train_other = Xtrain[:,56:]
test_word_match = Xtest[:,:48]
test_char_match = Xtest[:,49:55]
test_other = Xtest[:,56:]
'''


"""
print (Xtrain.shape[0])
print ("XTrain ")
print (beta_initial.shape)
print (ytrain.shape)
print(Xtrain.shape)

print (np.dot(Xtrain.T, ytrain).shape)
print (2*reg*beta_initial.shape)
"""

def mu(beta, x):
    beta = beta.reshape((57,1))
    x = x.reshape((57,1))
    dot = np.dot(beta.T, x)
    denom = 1 + np.exp(-dot[0])
    return 1 / denom

def mu_vector(x, beta):
    mu_vec = []
    #mu is probabily of (Y|X)
    for i in range(x.shape[0]):
        mu_vec.append(mu(beta, x[i]))
    #print ("Finishing Mu vector")
    return np.array(mu_vec)
    
#mu_vec = mu_vector(Xtrain, beta_initial)
#print (mu_vec)

def loss_function(x,y,u,beta,reg):
    total = 0
    for i in range(y.shape[0]):
        first = np.multiply(y[i],np.log(u[i]))
        second = np.multiply((1-y[i]),np.log(1- u[i]))
        total += (first[0] + second[0])
    l2_norm = 0
    for i in range(beta.shape[0]):
        l2_norm += pow(beta[i], 2)
    l2_norm = np.sqrt(l2_norm)

    return l2_norm - total
        
#loss = loss_function(Xtrain, ytrain, mu_vec, beta_initial, reg)
#print (str(loss) + " is the loss")

def gradient(reg, beta, x, y, u):
    total = 0
    for i in range(x.shape[0]):
        total += x[i] * (y[i] - u[i])
    return (2.0*reg*beta - total)

#print (gradient(reg, beta_initial, Xtrain, ytrain, mu_vec))
        
def gradient_descent(old_beta, eta, reg, x, y, mu):
    #Gradient vector of size 57
    grad = gradient(reg, old_beta, x, y, mu)
    new_beta = old_beta - eta * grad
    #return size 57
    return new_beta    

    

"""
new_beta = gradient_descent(beta_initial, 1, reg, Xtrain, ytrain, mu_vec)
print (new_beta.shape)
print (beta_initial)
print (new_beta)
mu_vec = mu_vector(Xtrain, new_beta)
next_beta = gradient_descent(new_beta, 1, reg, Xtrain, ytrain, mu_vec)
print (next_beta)
"""

def batch_iterations(num, beta_initial, Xtrain, ytrain, reg):
    beta = beta_initial
    for x in range(num):
        #print ("Iteration number " + str(x+1))
        mu_vec = mu_vector(Xtrain, beta)
        loss = loss_function(Xtrain, ytrain, mu_vec, beta, reg)
        #print (str(loss) + " is the loss")
        g_loss.append(loss)
        beta = gradient_descent(beta, .0001, reg, Xtrain, ytrain, mu_vec)
        #print (beta.shape)
        #print (beta)
        
def stochastic(i,x, y, beta):
    mu_vec = mu_vector(Xtrain, beta)
    mui = mu_vec[i]
    xi = Xtrain[i]
    yi = y[i]
    gi = (yi - mui) * xi
    return gi
    
#stochastic(1, Xtrain, ytrain, beta_initial)
    
def stochastic_gradient(x, y, beta, eta):
    rand_idx = rand.randint(1, x.shape[0]-1)
    #print ("RANDOM NUMER IS " + str(rand_idx))
    stoc_beta = stochastic(rand_idx, x, y, beta)  
    update = eta * stoc_beta
    new_beta = beta + update
    
    return new_beta
    
#stochastic_gradient(Xtrain, ytrain, beta_initial, .001)
def stochastic_iterations(num, x, y, beta, eta, reg):
    for i in range(num):
        mu_vec = mu_vector(x, beta)
        loss = loss_function(x, y, mu_vec, beta, reg)
        #print (str(loss) + " is the loss")
        s_loss.append(loss)
        beta = stochastic_gradient(x, y, beta, eta)
        
        
# Decreasing learning 
def dec_stochastic_iterations(num, x, y, beta, eta, reg):
    for i in range(num):
        mu_vec = mu_vector(x, beta)
        loss = loss_function(x, y, mu_vec, beta, reg)
        rand_idx = rand.randint(1, x.shape[0]-1)
        #print ("RANDOM NUMER IS " + str(rand_idx))
        #print (str(loss) + " is the loss")
        dec_loss.append(loss)
        
        stoc_beta = stochastic(rand_idx, x, y, beta)  
        eta = 1.0/float(i+1.0)
        update = eta * stoc_beta
        beta = beta + update        
        
#i) Normalize the features to have 0 mean and unit variance 

#Gradient = summation of 1 to n yi - mu_i (Beta(t))) Xi
#Normalization
print ("4i) Normalize Features with 0 mean and unit variance ========================")



def standardize(training_points):
    for x in range(training_points.shape[1]):
        mean = np.mean(training_points[:,x])
        std = np.std(training_points[:,x])
        training_points[:,x] -= mean
        training_points[:,x] = np.divide(training_points[:,x], float(std))  
    return training_points
    

Xtrain = standardize(train)

#regularization parameter
reg = 1
g_loss = []
s_loss = []
dec_loss = []
beta_initial = np.zeros(Xtrain.shape[1])

print ("Gradient Descent")      
batch_iterations(50, beta_initial, Xtrain, ytrain, reg)  

plt.plot(np.arange(len(g_loss)), np.array(g_loss), 'ro')
plt.axis([0, len(g_loss), min(g_loss), max(g_loss)])
plt.title('4i-1')
plt.show()
     
print ("Stochastic Descent")

stochastic_iterations(50, Xtrain, ytrain, beta_initial, .001, reg)

plt.plot(np.arange(len(s_loss)), np.array(s_loss), 'ro')
plt.axis([0, len(s_loss), min(s_loss), max(s_loss)])
plt.title('4i-2')
plt.show()

print("Decreasing Learning Rate")
dec_stochastic_iterations(400, Xtrain, ytrain, beta_initial, .001, reg)

plt.plot(np.arange(len(dec_loss)), np.array(dec_loss), 'ro')
plt.axis([0, len(dec_loss), min(dec_loss), max(dec_loss)])
plt.title('4i-3')
plt.show()
#####Plotting three graphs


g_loss = []
s_loss = []
dec_loss = []

#ii) Transform log(Xij +0.1)
print ("4ii) Transform log(xij + 0.1) ========================")
#Remove the Nan's
daSet = io.loadmat('spam.mat')
datrain  = daSet['Xtrain']
def transform(training_points):
    for x in range(training_points.shape[1]):
        training_points[:,x] = np.log(training_points[:,x] + 0.1)
    return training_points
log_train = transform(datrain)
    

#print (log_train)
print ("Gradient Descent")      
batch_iterations(200, beta_initial, log_train, ytrain, reg)   

plt.plot(np.arange(len(g_loss)), np.array(g_loss), 'ro')
plt.axis([0, len(g_loss), min(g_loss), max(g_loss)])
plt.title('4ii-1')
plt.show()
     

    
print ("Stochastic Descent")
stochastic_iterations(200, log_train, ytrain, beta_initial, .001, reg)

plt.plot(np.arange(len(s_loss)), np.array(s_loss), 'ro')
plt.axis([0, len(s_loss), min(s_loss), max(s_loss)])
plt.title('4ii-2')
plt.show()


print("Decreasing Learning Rate")
dec_stochastic_iterations(200, log_train, ytrain, beta_initial, .001, reg)

plt.plot(np.arange(len(dec_loss)), np.array(dec_loss), 'ro')
plt.axis([0, len(dec_loss), min(dec_loss), max(dec_loss)])
plt.title('4iii-3')
plt.show()


#iii) Binarize the features 

g_loss = []
s_loss = []
dec_loss = []


print ("4iii) binarize the features ========================")
dSet = io.loadmat('spam.mat')
newtrain  = dSet['Xtrain']
def binarize(training_points):
    for y in range(training_points.shape[1]):
        for x in range(training_points.shape[0]):

            training_points[x,y] = 1 if training_points[x,y] > 0 else 0
            
    return training_points
            
binaryTrain = binarize(newtrain)

#print (log_train)
print ("Gradient Descent")      
batch_iterations(200, beta_initial, binaryTrain, ytrain, reg)     

plt.plot(np.arange(len(g_loss)), np.array(g_loss), 'ro')
plt.axis([0, len(g_loss), min(g_loss), max(g_loss)])
plt.title('4iii-1')
plt.show()

  
print ("Stochastic Descent")
stochastic_iterations(200, binaryTrain, ytrain, beta_initial, .001, reg)

plt.plot(np.arange(len(s_loss)), np.array(s_loss), 'ro')
plt.axis([0, len(s_loss), min(s_loss), max(s_loss)])
plt.title('4iii-2')
plt.show()


print("Decreasing Learning Rate")
dec_stochastic_iterations(200, binaryTrain, ytrain, beta_initial, .001, reg)

plt.plot(np.arange(len(dec_loss)), np.array(dec_loss), 'ro')
plt.axis([0, len(dec_loss), min(dec_loss), max(dec_loss)])
plt.title('4iii-3')
plt.show()



"""
48 features giving the percentage (0 - 100) of words in a given message which match a given word on the list. The list contains words such as business, free, george, etc. (The data was collected by George Forman, so his name occurs quite a lot!)
6 features giving the percentage (0 - 100) of characters in the email that match a given character on the list. The characters are ;( [ ! $ # .
Feature 55: The average length of an uninterrupted sequence of capital letters
Feature 56: The length of the longest uninterrupted sequence of capital letters
Feature 57: The sum of the lengths of uninterrupted sequence of capital letters
"""


"""
Visualizing Covariance Matrices

"""

import numpy as np

reg = 0.07

beta_initial = np.array([-2.0,1.0,0.0]).T
xData = np.array([[0.0,1.0,0.0,1.0],[3.0,3.0,1.0,1.0]]).T
yData = np.array([1.0,1.0,0.0,0.0]).T

xData = np.insert(xData, xData.shape[1], 1, axis=1)
#yData = np.insert(yData, yData.shape[1], 1, axis=1)


def mu(beta, x):
    beta = beta.reshape((3,1))
    x = x.reshape((3,1))
    dot = np.dot(beta.T, x)
    denom = 1 + np.exp(-dot[0])
    return 1 / denom
    
def mu_vector(x, beta):
    mu_vec = []
    #mu is probabily of (Y|X)
    for i in range(x.shape[0]):
        mu_vec.append(mu(beta, x[i]))
    return np.array(mu_vec)
    
def gradient(reg, beta, x, y, u):
    total = 0
    for i in range(x.shape[0]):
        total += x[i] * (y[i] - u[i])
    return (2.0*reg*beta - total)
    
def hessian(reg, x, u):
    total = 0
    for i in range(x.shape[0]):
        x_term = np.dot(x[i], x[i].T)
        mu_term = np.dot(u[i], (1 - u[i]))
        total -= np.dot(x_term, mu_term)
        
    return 2*reg + total

def update(beta, x, y, reg):
    for l in range(3):
        mu_vec = mu_vector(x, beta)  
        print ("Mu" + str(l) + ":" )
        print  mu_vec
        hes = hessian(reg, x, mu_vec)  
        grad = gradient(reg, beta, x, y, mu_vec)
        up = 1/hes * grad
        beta += up
        print ("Beta" + str(l) + ": ")
        print (beta)
    

update(beta_initial, xData, yData, reg)