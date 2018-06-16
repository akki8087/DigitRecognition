# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 20:11:08 2018

@author: NP
"""

import pandas as pd
import numpy as np

data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X = data.iloc[:,1:]
y = data.iloc[:,0]


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
'''
X_train = X
y_train = y
X_test = test
'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(classifier.score(X_test,y_test))
'''
result = pd.DataFrame()

result['ImageId'] = range(1,len(test)+1)

result['Label'] = y_pred

result.to_csv('DR_NB1.csv',index = False)

'''