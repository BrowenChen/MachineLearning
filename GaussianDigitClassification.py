# -*- coding: utf-8 -*-
"""
Problem 4 Homework 3
"""
import numpy as np
import math
import scipy 
import csv
from scipy import io
import sklearn
import random
from numpy import linalg as LA
from random import shuffle
from sklearn import svm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import csv
from scipy.stats import multivariate_normal
from numpy import matrix
from scipy.stats import norm
import matplotlib.mlab as mlab



#QUESTION 1

### PROBLEM 1 --- Visualizing Eigenvectors of Gaussian Covariance
print "Problem 1 ========================================"

mu, sigma = 3, 3 # mean and standard deviation
x1 = np.random.normal(mu, sigma, 100)
x2 = (.5) * x1 + np.random.normal(4,4,100)

final_vector = np.array([x1, x2])
#print (final_vector)
meanx = np.mean(final_vector[0,])
meany = np.mean(final_vector[1,])
print ("The mean: " + str(meanx) + " and " + str(meany) )

plt.plot(final_vector[0,],final_vector[1,],'ro')
plt.show()


#Covariance matrix of the sampled data
covariance_matrix = np.cov(final_vector[0,],final_vector[1,])
print (covariance_matrix)

#eignvalues and eigenvector
w,eig = LA.eig(covariance_matrix)

print (" ")
print ("Eigenvalues: ")
print (w)
print (" ")
print ("EigenVectors: ")
print (eig)

print ("part d")

plt.plot(x1, x2, 'b')
plt.axis([-15, 15, -15, 15])


print "Didn't finish parts d and e " 
"""
vector1x = eig[1][0][0] * eig[0][0]
vector1y = eig[1][1][0] * eig[0][0]
vector2x = eig[1][0][1] * eig[0][1]
vector2y = eig[1][1][1] * eig[0][1]

plt.arrow(meanx,meany,vector1x, vector1y, shape = 'full', lw = 2, length_includes_head = True, head_width = .05)
plt.arrow(meanx,meany,vector2x, vector2y, shape = 'full', lw = 2, length_includes_head = True, head_width = .05)

print ("part e")


x1[:] = [a - mean[0] for a in x1]
x2[:] = [b - mean[1] for b in x2]
rotation = eig[1]
rotation = np.transpose(rotation)
combined = np.vstack((x,y))
final = np.dot(rotation, combined)
plt.figure()
plt.plot(final_vector[0,], final_vector[1,], 'b')
plt.axis([-15,15,-15,15])
plt.show()

"""


#### PROBLEM 3 --- Isocontours of Normal Distributions 
print "Problem 3 ========================================"

print (" ")
print (" " ) 
print ( " 2-D Gaussian Isocontours " )


mu = np.array([1,1])
sigma = np.array([[2,0],[2,0]])

delta = 0.025
x = np.arange(-4.0, 5.0, delta)
y = np.arange(-4.0, 5.0, delta)
X, Y = np.meshgrid(x, y)

Z1 = mlab.bivariate_normal(X, Y,  2.0, 1.0, 1.0, 1.0, 0)
plt.figure()
CS = plt.contour(X, Y, Z1)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('3a) Isocontour')

mu = np.array([-1,2])
sigma = np.array([[3,1,1,2]])
cov = np.cov(sigma)

delta = 0.025
x = np.arange(-4.0, 5.0, delta)
y = np.arange(-4.0, 5.0, delta)
X, Y = np.meshgrid(x, y)

Z1 = mlab.bivariate_normal(X, Y, 3.0, 2.0, -1.0, 2.0,  cov)
plt.figure()
CS = plt.contour(X, Y, Z1)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('3b) Isocontour')

####################3c)
sigma = np.array([[1,1,1,2]])
cov = np.cov(sigma)

delta = 0.025
x = np.arange(-4.0, 5.0, delta)
y = np.arange(-4.0, 5.0, delta)
X, Y = np.meshgrid(x, y)

Z1 = mlab.bivariate_normal(X, Y, 3.0, 2.0, 0.0, 2.0, 1.0)
Z2 = mlab.bivariate_normal(X, Y, 3.0, 2.0, 2.0, 0.0,  1.0)

Z=Z1-Z2

plt.figure()
CS = plt.contour(X, Y, Z1)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('3c) Isocontour')


####################3d)
#sigma = np.array([[1,1,1,2]])
#cov = np.cov(sigma)

delta = 0.025
x = np.arange(-4.0, 5.0, delta)
y = np.arange(-4.0, 5.0, delta)
X, Y = np.meshgrid(x, y)

Z1 = mlab.bivariate_normal(X, Y, 1.0, 2.0, 0.0, 2.0, 1.0)

Z2 = mlab.bivariate_normal(X, Y, 3.0, 2.0, 2.0, 0.0,  1.0)
Z=Z1-Z2

plt.figure()
CS = plt.contour(X, Y, Z1)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('3d) Isocontour')



####################3e)
#sigma = np.array([[1,1,1,2]])
#cov = np.cov(sigma)

delta = 0.025
x = np.arange(-4.0, 5.0, delta)
y = np.arange(-4.0, 5.0, delta)
X, Y = np.meshgrid(x, y)

Z1 = mlab.bivariate_normal(X, Y, 1.0, 2.0, 1.0, 1.0,  0.0)

Z2 = mlab.bivariate_normal(X, Y, 2.0, 2.0,-1.0, -1.0,  1.0)
Z=Z1-Z2

plt.figure()
CS = plt.contour(X, Y, Z1)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('3e) Isocontour')




####PROBLEM 4################




#Normalize by dividing pixel values by l2 norm of pixel values 

#Shaping the data
imageDataset = io.loadmat('data/digit-dataset/train.mat')
d_imgs = numpy.reshape(imageDataset['train_image'], [784, 60000])
d_imgs_transformed = [d_imgs[:, img] for img in range(60000)]
d_labels = imageDataset['train_label']
        
##########4a) MLE to find sample mean and covariance matrix
        
#For each digit class, fit a mean vector and covariance
# 0 - digit, take all the features of 0 data point. Compute mean vector

#The indices for each digit class
digitLabelIndices = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
idx = 0
for x in d_labels.T[0]:
    digitLabelIndices[x].append(idx)
    idx+= 1
    
digit_class_imgs = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}

#L2 Normalization 
#summation of pow(x,2)and square root
print("Normalizing ")
norm_sum = 0
for x in d_imgs_transformed[0]:
    norm_sum += pow(x, 2)
l2_norm = math.sqrt(norm_sum)

#norm_sum = np.sum(np.power(np.power(d_imgs_transformed[0], 2)))
#l2_norm = np.sqrt(norm_sum)

#print(l2_norm)

print("Normalizing matrix with L2 norm ")

#print (d_imgs_transformed[0]/l2_norm)


for key in digit_class_imgs:
    for i in digitLabelIndices[key]:
        digit_class_imgs[key].append(d_imgs_transformed[i]/l2_norm)

#Design Matrices
design_matrices = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
for x in design_matrices:
    design_matrices[x] = np.array(digit_class_imgs[x]).T

#Mean Vector
mean_vectors ={0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}

#Find the average of all the 784 pixel values in each 
for y in range(10):
    d_class = np.array(digit_class_imgs[y])
    for x in range(784):
        mean_vectors[y].append(np.mean(d_class[:,x]))

#Covariance Matrix
covariance_matrices = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
for x in covariance_matrices:
    covariance_matrices[x] = np.cov(design_matrices[x])


#Why are these unbiased?

##########2b) Prior distribution is #of items of a digit class / Total items
print (" Computing Prior distributions of each digit class  ")
print (" ")
print (" Prior distributions is # of labels / total occurences of that digit")
priorDistributions = {}
total_labels = float(len(d_labels))
for x in range(10):
    occurrences=float(len(digitLabelIndices[x]))
    prior = occurrences/total_labels
    priorDistributions[x] = prior

##########4c) Plot a covariance Matrix
print(" Plotting a Covariance matrix for digit: 0 " )

data = covariance_matrices[0]
plt.matshow(data)
plt.title('Covariance Matrix')
plt.colorbar()
plt.ylabel('T ')
plt.xlabel(' ')
plt.show()

#plt.plot(x,y,'x'); plt.axis('equal'); plt.show()


##########4d) Classify digits on the basis of posterior probabilities
print (" Classifying ")
print "Problem 4 ========================================"

imgTest = io.loadmat('data/digit-dataset/test.mat')
imgTestFlattened = []
for i in range(5000):
    imgTestFlattened.append(np.ravel(imgTest['test_image'][:,:,i]))
imgLabels = imgTest['test_label']
test_Dictionary = {} #(img, label)
for i in range(5000):
    test_Dictionary[i] = (imgTestFlattened[i], imgLabels[i][0]) #Weird index format

#training_points = [100, 200, 500, 1000, 2000, 5000, 10000, 30000, 60000]
training_points = [50] #Temporary to speed things up 

 
#model class conditionals as Guassians N(ui, Sigma_overall)
#Create PDF for Gaussian Class conditional. 

#4di)
print ("Starting part 4dI")
#Sigma_overall to be the average covariance matrix.
#Need something PSD to use the function 

c_a = np.array(covariance_matrices.values())
sigma_overall = np.zeros((784,784))
for x in c_a:
    sigma_overall=np.add(sigma_overall,x)    
sigma_overall = sigma_overall/10
#This is a singular matrix, need to add identity matrix

#Sigma_overall + alpha*identity matrix 
#print (" Is this covariance matrix invertible? ")
identity_matrix = np.identity(784)
alpha = pow(10,-2)
print ("Alpha value: "+str(alpha)  )
sigma_overall = sigma_overall + identity_matrix*alpha

#print ("Checking if Covariance Overall is invertible ")
#B = LA.inv(sigma_overall)



#Argmax i P(x|di=x)*p(x)
def class_posterior_prob(x, sigma_overall):
    maxClassConditional = float("-inf")
    digit = None 
    #Find argmax x P(y=i|X)
    for i in digit_class_imgs:
        pdf = multivariate_normal.logpdf(x, mean_vectors[i], sigma_overall)
        prior = priorDistributions[i]
        #Log likelihood add
        classCond = pdf+prior
        if maxClassConditional < classCond:
            maxClassConditional=classCond
            digit=i
    return digit
    
for setSize in training_points:
    keys = test_Dictionary.keys()
    shuffle(keys)
    outComes = []
    trueLabels = []
    correct = 0.0
    total = 0.0
    

    for key in keys[:setSize]:
        #print test_Dictionary[key][0]
        digit_class = class_posterior_prob(test_Dictionary[key][0],sigma_overall)
        trueLabels.append(test_Dictionary[key][1])   
        if digit_class == test_Dictionary[key][1]:
            correct += 1.0
        total += 1.0    
        outComes.append(digit_class)
    
    #print (outComes)  
    #print (trueLabels)
    print ("Success rate is for " + str(setSize) + " training samples is : " + str(correct/total))
    print ("Error rate : " + str(1 - correct/total))
      

#4dii)
#Using i mean vector and i Coveriance matrix 
print ("Starting part 4sII)") 

def class_posterior_prob_ii(x):
    maxClassConditional = float("-inf")
    digit = None 
    for i in digit_class_imgs:
        pdf = multivariate_normal.logpdf(x, mean_vectors[i], covariance_matrices[i])
        prior = priorDistributions[i]
        classCond = pdf+prior
        if maxClassConditional < classCond:
            maxClassConditional=classCond
            digit=i
    return digit

for setSize in training_points:
    
    keys = test_Dictionary.keys()
    shuffle(keys)
    outComes = []
    trueLabels = []
    
    correct = 0.0
    total = 0.0    
    
    #Change this 
    for key in keys[:setSize]:
        #print test_Dictionary[key][0]
        digit_class = class_posterior_prob_ii(test_Dictionary[key][0])
        trueLabels.append(test_Dictionary[key][1])
        if digit_class == test_Dictionary[key][1]:
            correct += 1.0
        total += 1.0
        outComes.append(digit_class)
        
    
    #print (outComes)
    #print (trueLabels)
    print ("Success rate is for " + str(setSize) + " training samples is : " + str(correct/total))
    print ("Error rate : " + str(1 - correct/total))


########################Cross validation to train values of alpha


########################
#Train X amount of data points and plot error rate
#for training_size in training_points:
#    for set_size in digits[:training_size]:

#fig1 = plt.figure()
#ax = fig1.add_subplot(111)
#ax.plot(x)

#fig, ax = plt.subplots(1, 1)
#x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
#norm.pdf(x)
#ax.plot(x, norm.pdf(x),'r-', lw=5, alpha=0.6, label='norm pdf')
#.logPdf function 


#####KAGGLE COMPETITION

print("KAGGLE DIGITS COMPETITION")

imgKaggle = io.loadmat('data/digit-dataset/kaggle.mat')
imgKaggleFlattened = []
label_Outcome = []

#5000 for full kaggle 
for i in range(3):
    imgKaggleFlattened.append(np.ravel(imgKaggle['kaggle_image'][:,:,i]))
#kaggleLabels = imgKaggle['kaggle_label']
##kaggle_Dictionary = {} #(img, label)
#print imgKaggleFlattened

for x in imgKaggleFlattened:
    label_Outcome.append(class_posterior_prob_ii(x))
print label_Outcome

print("Writing to kaggle submission")

with open('kaggle_email_submission.csv', 'wb') as csvfile:
	predictionWriter = csv.writer(csvfile, delimiter=",")
	predictionWriter.writerow(['Id','Category'])
	predictionId = 1
	for label in label_Outcome:
         predictionWriter.writerow([predictionId, label])
         predictionId+=1




print("KAGGLE SPAM COMPETITION")

spamKaggle = io.loadmat('data/spam-dataset/spam_data.mat')
spamKaggleTrainingFlattened= []
label_Outcome = []

spam_labels = spamKaggle['training_labels']
spam_training = spamKaggle['training_data']
spam_test = spamKaggle['test_data']



print (" SPAM TRAINING ")

#Training the spam classes
#The indices for each digit class
spamLabels = {0:[], 1:[]}

idx = 0

for x in spam_labels[0]:
    spamLabels[x].append(idx)
    idx+= 1
    

spam_class = {0:[],1:[]}

for key in spam_class:
    for i in spamLabels[key]:
        spam_class[key].append(spam_training[i])
        
"""
#L2 Normalization for spam.  
#summation of pow(x,2)and square root
print("Normalizing ")
norm_sum = 0
for x in spam_class[0][0]:
    norm_sum += pow(x, 2)
l2_norm = math.sqrt(norm_sum)

print spam_class[0][0]
print(l2_norm)
print("Normalizing matrix with L2 norm ")

for key in spam_classes:
    for i in spamLabels[key]:
        spam_classes[key].append(spam_transformed[i]/l2_norm)

"""


#Design Matrices
s_design_matrices = {0:[],1:[]}
for x in s_design_matrices:
    s_design_matrices[x] = np.array(spam_training[x]).T


#Mean Vector
spam_m_vectors ={0:[],1:[]}

#Find the average of all the 32 pixel values in each 
for y in range(2):
    s_class = np.array(spam_class[y])
    for x in range(32):
        spam_m_vectors[y].append(np.mean(s_class[x]))

#Spam Covariance Matrix
s_covariance_matrices = {0:[],1:[]}
for x in s_covariance_matrices:
    s_covariance_matrices[x] = np.cov(s_design_matrices[x])


print (" Computed mean vectors and covariance matrices for each digit class. ")

#Train the spam dataset
#kaggleLabels = imgKaggle['kaggle_label']
##kaggle_Dictionary = {} #(img, label)
#print imgKaggleFlattened
s_priorDistributions={0:[], 1:[]}

s_priorDistributions[0] = (float(len(spamLabels[0]))) / 5172.0
s_priorDistributions[1] = (float(len(spamLabels[1]))) / 5172.0


def spam_class_posterior_prob(x):
    mClassConditional = float("-inf")
    spam = None 
    for i in spam_class:
        pdf = multivariate_normal.logpdf(x, spam_m_vectors[i], s_covariance_matrices[i])
        prior = s_priorDistributions[i]
        classCond = pdf+prior
        if mClassConditional < classCond:
            mClassConditional=classCond
            spam=i
    return spam


spam_label_outcome = []
for x in spam_test:
    spam_label_outcome.append(spam_class_posterior_prob(x))
    

print("Writing to kaggle spam submission")

with open('kaggle_spam_submission.csv', 'wb') as csvfile:
	predictionWriter = csv.writer(csvfile, delimiter=",")
	predictionWriter.writerow(['Id','Category'])
	predictionId = 1
	for label in spam_label_outcome:
         predictionWriter.writerow([predictionId, label])
         predictionId+=1
         
