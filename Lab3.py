import numpy as np
import os
os.environ["PATH"] += os.pathsep + r'D:\Codes\Environments\graphviz-2.38\bin'

from sklearn import datasets, metrics
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus

# data
iris = datasets.load_iris()
X = iris.data
y = iris.target


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


dtc = DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=5)
dtc.fit(X, y)

# max_leaf_nodes
dot_data = export_graphviz(dtc, out_file=None,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)

with open("dtc.png", "wb") as png:
    png.write(graph.create_png())


# In[]


