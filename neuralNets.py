# -*- coding: utf-8 -*-
"""
Neural Nets
3-Layer Network

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



digitsTrainingData = scipy.io.loadmat('digit-dataset/train.mat')
digitsTestData = scipy.io.loadmat('digit-dataset/test-2.mat')

test_images = digitsTestData['test_images']
test_images_flattened = [test_images[:,:,i].flatten()/255.0 for i in range(test_images.shape[2])]

train_labels = digitsTrainingData['train_labels']
#Testing normalization visualization 
t_img = digitsTrainingData['train_images']
t_img_norm = np.array([(t_img[:,:,i]/ 255.0 ) for i in range(t_img.shape[2])])

training_imgs = [t_img_norm[i].flatten() for i in range(t_img_norm.shape[0])]

'''
Shuffle the array
'''
keys = np.arange(60000)
random.shuffle(keys)
shuffled_labels = []
shuffled_images = []

validation_labels = []
validation_images = []
for i in range(60000):
    if i < 50000:
        shuffled_images.append(training_imgs[keys[i]])
        shuffled_labels.append(train_labels[keys[i]])
    else:
        validation_images.append(training_imgs[keys[i]])
        validation_labels.append(train_labels[keys[i]])
    
    
def trainNeuralNetwork(images, labels, eta, loss_function, *args):
    stopping_criteria = 0
    
    
    w_1 = np.random.normal(0.0, 0.001, (785, 200)) # (785x200)
    s = np.random.normal(0, 1, (785, 200))
    w_2 = np.random.normal(0.0, 0.001, (201,10)) #(201x10)
    
    loss = []
    
    #Change stopping criteria later. 
    while (stopping_criteria < 500000):
        random_idx = randint(1, 49999)
        label = labels[random_idx][0]
        
        #Changing labels to 10 classes each [1,0,0,0...]
        datum_label = np.zeros(10)
        datum_label[label] = 1 #size (10x1)
        
        data_point = (images[random_idx], datum_label)
        training = np.append(data_point[0], [1]) #Size (785x1)
        assert(training.shape == (785,))
                
        #Forward pass  
        z1 = np.dot(training,w_1) #z1
        a1 = np.append(np.tanh(z1), [1])#Size (201x1)
        assert(a1.shape == (201,))
        z2 = np.dot(a1, w_2)
        y_out = expit(z2)#Size (10x1)
        assert(y_out.shape == (10,))

        #Backpropogation. 

        #Mean Squared error derivative        
        #error_deriv = mean_squared_deriv(y_out, data_point[1])
        #Cross-entropy error derivative 
        error_deriv = loss_function(y_out, data_point[1])
        
        sigmoid_deriv = np.multiply(y_out, (1-y_out))
        delta_out = np.multiply(error_deriv, sigmoid_deriv) #Size (10x1)

        assert(delta_out.shape == (10,))
        
        w2Delta_out = np.dot(w_2, delta_out)
        tanh_deriv = np.subtract(1, np.power(a1,2))
        delta_hidden = np.multiply(w2Delta_out, tanh_deriv)
        
        assert(delta_hidden.shape == (201,))
        
        w2_derivative = np.outer(delta_out, a1).T #Size (201x10)
        assert(w2_derivative.shape == (201,10))        
        w1_derivative = np.outer(training, delta_hidden) #Size (785, 201)
        assert(w1_derivative.shape == (785,201))        
        
        w_2 -= np.multiply(eta,w2_derivative)
        #Remove the last element of the 201
        #print ("DELETING THE LAST COLUMN")
        w1_derivative= np.delete(w1_derivative, 200, axis=1)
        w_1 -= np.multiply(eta,w1_derivative)
        stopping_criteria += 1
        #print("Iteration " + str(stopping_criteria))
        if (stopping_criteria % 1000 == 0):
            print("Iteration " + str(stopping_criteria))
        
        if (stopping_criteria % 10000 == 0):
            l = calculate_acc(w_1, w_2)
            print("Iteration " + str(stopping_criteria) + " with accuracy " + str(l*100) + "%")
            loss.append(l)
            
    plt.plot(loss)
    plt.title("Accuracy each 10,000 Iterations")
    return (w_1, w_2)
    
def predictNeuralNetworks(weights, images):
    #Forward pass
    predicted = []
    for i in range(10000):
        z1 = np.dot(np.append(images[i], [1]),weights[0]) #z1
        a1 = np.append(np.tanh(z1), [1])#Size (201x1)
        z2 = np.dot(a1, weights[1])
        y_out = expit(z2)#Size (10x1)        
        predicted.append(np.argwhere(y_out.max() == y_out)[0][0])
    return predicted

def calculate_acc(w_1, w_2):
    #Forward pass  
    loss = []
    predicted = []
    for i in range(10000):
        z1 = np.dot(np.append(validation_images[i], [1]),w_1) #z1
        a1 = np.append(np.tanh(z1), [1])#Size (201x1)
        z2 = np.dot(a1, w_2)
        y_out = expit(z2)#Size (10x1)
        
        ##ACCURACY CHECKER IS CORRECT
        predicted.append(np.argwhere(y_out.max() == y_out)[0][0])
    #print (predicted)        
    return accuracy_score(validation_labels, predicted)



def mean_square_loss(labels, output):
    return np.sum((.5) * np.power(np.subtract(labels,output), 2))

def cross_entropy_error(y, h_x):
    return -(np.dot(y,np.log(h_x)) + (1-y), np.log(1-h_2))


def mean_squared_deriv(y_out, true_label):
    return np.subtract(y_out, true_label)

def cross_entropy_deriv(y_out, true_labels):
    return np.multiply((1-true_labels), y_out) - np.multiply(true_labels, (1-y_out)) 
    
    
    
#Choose loss function, mean_squared_deriv or cross_entropy_deriv
weights = trainNeuralNetwork(np.array(shuffled_images),np.array(shuffled_labels),.00725, cross_entropy_deriv)

loss =  calculate_acc(weights[0], weights[1])
print("Correct classifications " + str(loss*100) + "%")

predictions = predictNeuralNetworks(weights, test_images_flattened)
print(predictions)



print("Writing to kaggle spam submission")

with open('kaggle_FIN_submission.csv', 'wb') as csvfile:
	predictionWriter = csv.writer(csvfile, delimiter=",")
	predictionWriter.writerow(['Id','Category'])
	predictionId = 1
	for label in predictions:
         predictionWriter.writerow([predictionId, label])
         predictionId+=1


'''
print ("Testing mean squre loss")
labels = [1,2,3,4,5,6,7,8,9]
output = [2,3,4,5,6,7,8,9,10]
print (mean_square_loss(labels, output))
'''

'''
Bugs 4/22

Not working

- changed tanh(a1) to a1
- changed dot to multiply 

- Classifying as all ones, forward pass works, 
- Shuffled dataset
- Normalized dividing by 255.0
- Normalizing with std deviation produces nan
- 1000 iterations 17% accuracy

Working
- forward pass
- Accuracy 
'''