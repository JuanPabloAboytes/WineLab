# USE IT IN COLLABS, select de csv dataset.
# Juan Pablo Aboytes Novoa A01701249
from google.colab import files
uploaded = files.upload()

#import all we need
import pandas as pd
import numpy as np
import os

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt

#dataset.
df = pd.read_csv("winequality-red.csv")
#check the first 5
df.head()

#split the x's and y's
inputs = df.drop('quality', axis='columns')
target = df['quality']

inputs
#all are numerical values, i do not need preprocess anything

from sklearn import tree
#model = DecisionTreeClassifier(max_depth = 3)
model = tree.DecisionTreeClassifier()
model.fit(inputs,target)

#model.score(inputs,target)
#predictions.
# 7.2,0.66,0.33,2.5,0.068,34.0,102.0,0.9941399999999999,3.27,0.78,12.8,         6
# 6.6,0.725,0.2,7.8,0.073,29.0,79.0,0.9977,3.29,0.54,9.2,                       5
# 6.0,0.31,0.47,3.6,0.067,18.0,42.0,0.99549,3.39,0.66,11.0,                     6

#model.predict([[7.2,0.66,0.33,2.5,0.068,34.0,102.0,0.9941399999999999,3.27,0.78,12.8]])
#model.predict([[6.6,0.725,0.2,7.8,0.073,29.0,79.0,0.9977,3.29,0.54,9.2]])
model.predict([[6.0,0.31,0.47,3.6,0.067,18.0,42.0,0.99549,3.39,0.66,11.0]])

#split the x's and y's labels
feature_df = df.columns[:11]
target_df = df.columns[11]

from sklearn import tree
import graphviz

# dot is a graph description language
dot = tree.export_graphviz(model, out_file=None, 
                           feature_names=feature_df,
                           class_names=target_df,
                           filled=True, rounded=True,  
                           special_characters=True) 

# we create a graph from dot source using graphviz.Source
graph = graphviz.Source(dot) 
graph
