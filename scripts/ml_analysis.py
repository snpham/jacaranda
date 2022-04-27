import numpy as np
from astropy.io import fits
import pandas as pd
import crossmatch
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression


def randomforest(max_depth):
    # initialize model
    clf = RandomForestClassifier(max_depth=max_depth, random_state=42)

    # train the model
    clf = clf.fit(X_train, Y_train)

    # make predictions using the same features
    predictions = clf.predict(x_test)
    f1 = f1_score(y_test, predictions, average='macro')
    print(f'randomforest ({max_depth}) f1: {f1}')


def svms(kernel):

    # initialize model
    svc = svm.SVC(kernel=kernel)
    # train the model
    svc = svc.fit(X_train, Y_train)
    # make predictions using the same features
    predictions = svc.predict(x_test)
    f1 = f1_score(y_test, predictions, average='macro')
    print(f'svm ({kernel}) f1: {f1}')


if __name__ == '__main__':
    pass
    
    data = pd.read_csv('outputs/data_processed.csv', index_col=0, header=0)
    data = data.drop(['class', 'subclass'], 1)

    data['gz2class'] = data['gz2class'].str.replace('spiral', '0')
    data['gz2class'] = data['gz2class'].str.replace('elliptical', '1')
    data['gz2class'] = data['gz2class'].apply(pd.to_numeric)

    data_train = data.drop('gz2class', 1)
    data_label = data['gz2class'].values.tolist()

    X_train, x_test, Y_train, y_test = train_test_split(data_train, data_label, 
        test_size=0.2, random_state=42)

    # random forest
    maxdepth = [1, 5, 10, 15, 20, 30, 50]
    # for depth in maxdepth:
    #     randomforest(depth)
    # randomforest (1) f1: 0.7251825174511961
    # randomforest (5) f1: 0.7803936832026334
    # randomforest (10) f1: 0.8067914735666953
    # randomforest (15) f1: 0.8170546634558619
    # randomforest (20) f1: 0.8189910358389592
    # randomforest (30) f1: 0.8202444232296995
    # randomforest (50) f1: 0.8201612103335587
    
    # svms
    kernels = ['linear', 'poly', 'rbf']
    # for kernel in kernels:
    #     svms(kernel)
    # svm (linear) f1: 0.3645528726989167
    # svm (poly) f1: 0.3645528726989167
    # svm (rbf) f1: 0.3645528726989167

