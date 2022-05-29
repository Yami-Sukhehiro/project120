from pandas.core.common import random_state
from google.colab import files
data_to_upload = files.upload()
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score
from io import StringIO
from IPython.display import Image
import pydotplus
from sklearn.tree import export_graphviz
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("income.csv")
X = df[["age","hours-per-week","education-num","capital-gain","capital-loss"]]
Y = df["income"]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
model = GaussianNB()
model.fit(x_train, y_train)
diaprediction = model.predict(x_test)
accuracy = accuracy_score(y_test, diaprediction)
print(accuracy)
