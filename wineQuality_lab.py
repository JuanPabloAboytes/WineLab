# USE IT IN COLLABS, select de csv dataset.
#the data set are the white variants of the Portuguese "Vinho Verde" wine 
#purpose let you know which physiochemical properties make a wine 'good'
#it is measured on scale 0 to 10. 0 = poor quality, 10 excelent quality
from google.colab import files
uploaded = files.upload()

#import all we need
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt

#dataset.
df = pd.read_csv("winequality-red.csv")
#check the first 5          1588
df.head()

df.isna().sum()
#not null values, only need to split x's and y's

#split the x's and y's
inputs = df.drop('quality', axis='columns') #x
target = df['quality']     #y

inputs
#all are numerical values, i do not need preprocess anything

from sklearn import tree
#model = DecisionTreeClassifier(max_depth = 3)
X_train, X_test, y_train, y_test = train_test_split(inputs,target, test_size=0.33)
model = tree.DecisionTreeClassifier(criterion="entropy", max_depth=8)  #max_depth=3
model.fit(inputs,target)
#model.fit(X_train,y_train)

#model.score(inputs,target)
#predictions.
# 7.2,0.66,0.33,2.5,0.068,34.0,102.0,0.9941399999999999,3.27,0.78,12.8,               6
# 6.6,0.725,0.2,7.8,0.073,29.0,79.0,0.9977,3.29,0.54,9.2,                             5
# 6.3,0.51,0.13,2.3,0.076,29.0,40.0,0.99574,3.42,0.75,11.0,                           6
# 6.0,0.31,0.47,3.6,0.067,18.0,42.0,0.99549,3.39,0.66,11.0,                           6
# 6.8,0.62,0.08,1.9,0.068,28.0,38.0,0.99651,3.42,0.82,9.5,                            6
# 5.9,0.55,0.1,2.2,0.062,39.0,51.0,0.9951200000000001,3.52,0.76,11.2,                 6
# 5.4,0.74,0.09,1.7,0.08900000000000001,16.0,26.0,0.9940200000000001,3.67,0.56,11.6,  6


model.predict([[7.2,0.66,0.33,2.5,0.068,34.0,102.0,0.9941399999999999,3.27,0.78,12.8],[6.6,0.725,0.2,7.8,0.073,29.0,79.0,0.9977,3.29,0.54,9.2],[6.3,0.51,0.13,2.3,0.076,29.0,40.0,0.99574,3.42,0.75,11.0],[6.0,0.31,0.47,3.6,0.067,18.0,42.0,0.99549,3.39,0.66,11.0], [6.8,0.62,0.08,1.9,0.068,28.0,38.0,0.99651,3.42,0.82,9.5],[5.9,0.55,0.1,2.2,0.062,39.0,51.0,0.9951200000000001,3.52,0.76,11.2],[5.4,0.74,0.09,1.7,0.08900000000000001,16.0,26.0,0.9940200000000001,3.67,0.56,11.6]])


#y_predicted = model.predict(X_test)
#y_predicted = model.predict(input)

#accuracy
#accuracy_score(y_test,y_predicted)*100

#split the x's and y's labels
feature_df = df.columns[:11]
target_df = df.columns[11]


from sklearn import tree
from sklearn.externals.six import StringIO  
from IPython.display import Image 
import pydotplus
import graphviz

dot_data = StringIO()
# dot is a graph description language
dot = tree.export_graphviz(model, out_file=dot_data, 
                           feature_names=feature_df,
                           class_names=['0','1','2','3','4','5','6','7','8','9','10'],
                           filled=True, rounded=True,  
                           special_characters=True) 

# we create a graph from dot source using graphviz.Source
graph = graphviz.Source(dot)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())
#graph
