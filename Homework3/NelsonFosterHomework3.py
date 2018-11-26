#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 12:42:36 2018

@author: Nelson Foster
DATS 6202
Machine Learning I 
Homework 3 | Decison Trees
"""

#importing initial packages

import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn import tree
import sklearn as sk 
import pydotplus 
import graphviz

from IPython.display import Image

from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


#read in dataset



pregnancy = pd.read_csv('pregnancy.csv')

#Exploratory Data Analysis

print (pregnancy)

pregnancy.describe()


 # Printing the dataswet shape
print ("Dataset Length: ", len(pregnancy))
print ("Dataset Shape: ", pregnancy.shape)


pregnancy.head()
pregnancy.tail()

list(pregnancy)

#create some data frames 

pregtest = pd.DataFrame(pregnancy['Pregnancy Test'])
birthcontrol = pd.DataFrame(pregnancy['Birth Control'])
FemHygiene = pd.DataFrame(pregnancy['Feminine Hygiene'])
FolicAcid = pd.DataFrame(pregnancy['Folic Acid'])
PrenatalV = pd.DataFrame(pregnancy['Prenatal Vitamins'])
PrenatalY = pd.DataFrame(pregnancy['Prenatal Yoga'])
BodyPillow = pd.DataFrame(pregnancy['Body Pillow'])
GingerAle = pd.DataFrame(pregnancy['Ginger Ale'])
SeaBands = pd.DataFrame(pregnancy['Sea Bands'])
StopBuyCigs = pd.DataFrame(pregnancy['Stopped buying ciggies'])
Cigarettes = pd.DataFrame(pregnancy['Cigarettes'])
StopSmoke = pd.DataFrame(pregnancy['Smoking Cessation'])
StopBuyWine = pd.DataFrame(pregnancy['Stopped buying wine'])
Wine = pd.DataFrame(pregnancy['Wine'])
MatClothes = pd.DataFrame(pregnancy['Maternity Clothes'])
Pregnant = pd.DataFrame(pregnancy['PREGNANT'])



#creating training and testing split


X = pregnancy.values[:,0:16]
Y = pregnancy.values[:,-1]
 
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)



#Gini Method


clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)


# Render our tree.
gini_data = tree.export_graphviz(
    clf_gini, out_file=None,
    feature_names=pregnancy.columns,
    class_names=['Not Pregnant', 'Pregnant'],
    filled=True
)




#Entropy Method



clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)



sk.tree.export_graphviz(clf_entropy)


# Render our tree.
entropy_data = tree.export_graphviz(
    clf_entropy, out_file=None,
    feature_names=pregnancy.columns,
    class_names=['Not Pregnant', 'Pregnant'],
    filled=True
)



####Fixes the image rendering problem
import os
import sys

def conda_fix(graph):
        path = os.path.join(sys.base_exec_prefix, "Library", "bin", "graphviz")
        paths = ("dot", "twopi", "neato", "circo", "fdp")
        paths = {p: os.path.join(path, "{}.exe".format(p)) for p in paths}
        graph.set_graphviz_executables(paths)
######

graph = pydotplus.graph_from_dot_data(entropy_data)
conda_fix(graph)
Image(graph.create_png())


####Note to Reviewer: when conda_fix was added, I got the following error: 
#InvocationException: GraphViz's executable 
#"/Users/nfoster06/anaconda/Library/bin/graphviz/dot.exe" is not a file or 
#doesn't exist, however when I excluded conda_fix, I got a different error:
#InvocationException: GraphViz's executables not found. I was unable to 
#reconcile this error to render the the decision tree plots before submission
#deadline
####

#Preictions

#Gini

y_pred = clf_gini.predict(X_test)
y_pred

#Entropy

y_pred_en = clf_entropy.predict(X_test)
y_pred_en

accuracy = accuracy_score(y_test,y_pred)*100

print(accuracy)

#ROC Curve

from sklearn.metrics import roc_curve, auc


pregpred = clf_entropy.predict(X)


#Here we see the confusion matrix
sk.metrics.confusion_matrix(Pregnant, pregpred)


#The roc_curve has three outputs fpr:false positive rate, tpr: true positive rate and thresholds
false_positive_rate, true_positive_rate, thresholds = roc_curve(Pregnant, pregpred) #Here we are developing variables for each output

roc_auc = auc(false_positive_rate, true_positive_rate)


plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

sk.metrics.roc_curve(Pregnant,pregpred) # Stand alone outside of the above process


                    





