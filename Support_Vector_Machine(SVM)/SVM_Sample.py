'''
Created on Mar 16, 2017

@author: Leo Zhong
'''
from sklearn import svm

X = [[2, 0], [1, 1], [2,3]] #data point
y = [0, 0, 1] #define type

#clf: classifier
clf = svm.SVC(kernel = 'linear') #get function
clf.fit(X, y) #set model

print (clf)

# get support vectors
print (clf.support_vectors_)

# get indices of support vectors
print (clf.support_ )

# get number of support vectors for each class
print (clf.n_support_ )