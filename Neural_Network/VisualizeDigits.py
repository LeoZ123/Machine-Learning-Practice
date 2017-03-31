'''
Created on Mar 30, 2017

@author: Leo Zhong
'''
from sklearn.datasets import load_digits
import pylab as pl
from nltk.app.nemo_app import images

digits = load_digits()
print (digits.data.shape)

pl.gray()
for index in range(0,10):
    pl.matshow(digits.images[index])
pl.show()