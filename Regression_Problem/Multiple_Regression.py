'''
Created on Apr 16, 2017

@author: Leo Zhong
'''
from numpy import genfromtxt
import numpy as np
from sklearn import datasets, linear_model

dataPath = r"F:\MachineLearning\Data\MultiRegression.csv"
testData = genfromtxt(dataPath, delimiter=',')

print ("data")
print (testData)

X = testData[:, :-1]
Y = testData[:, -1]

print ("X:")
print (X)
print ("Y: ")
print (Y)

regr = linear_model.LinearRegression()

regr.fit(X, Y)

print ("coefficients")
print (regr.coef_)
print ("intercept: ")
print (regr.intercept_)

xPred = [102, 6]
yPred = regr.predict(xPred)

print ("predicted y: ")
print (yPred)



