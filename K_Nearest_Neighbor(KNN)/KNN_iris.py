'''
Created on Mar 7, 2017

@author: Leo Zhong
'''
from sklearn import neighbors
from sklearn import datasets

#Get KNN Classifier
knn = neighbors.KNeighborsClassifier()

#Get iris data set
iris = datasets.load_iris()

print (iris)

#Training Data
#iris.data: eigenvector formed by data, iris.target: target name
knn.fit(iris.data, iris.target)

#Perdict data 
predictedLabel = knn.predict([[0.1, 0.2, 0.3, 0.4]])

print (predictedLabel)
