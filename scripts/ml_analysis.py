import numpy as np
from astropy.io import fits
import pandas as pd
import crossmatch
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from itertools import product
import multiprocessing as mp
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import time
from sklearn.preprocessing import StandardScaler


def randomforest(max_depth):
    start = time.perf_counter()

    X_train, x_test, Y_train, y_test = load_sets()
    
    # initialize model
    clf = RandomForestClassifier(max_depth=max_depth, random_state=42)

    # train the model
    clf = clf.fit(X_train, Y_train)

    # make predictions using the same features
    predictions = clf.predict(x_test)
    f1 = f1_score(y_test, predictions, average='macro')
    time_s = time.perf_counter() - start
    print(f'randomforest ({max_depth}) f1: {f1:.4f}, time (s): {time_s:.2f}')


def svms(kernel):
    start = time.perf_counter()

    X_train, x_test, Y_train, y_test = load_sets()

    # initialize model
    svc = svm.SVC(kernel=kernel)
    # train the model
    svc = svc.fit(X_train, Y_train)
    # make predictions using the same features
    predictions = svc.predict(x_test)
    f1 = f1_score(y_test, predictions, average='macro')
    time_s = time.perf_counter() - start
    print('svm ({kernel}) f1: {f1:.4f}, time (s): {time_s:.2f}')


def logregression(penalty):
    start = time.perf_counter()

    X_train, x_test, Y_train, y_test = load_sets()

    logreg = LogisticRegression(penalty=penalty)
    logreg.fit(X_train, Y_train)
    predictions = logreg.predict(x_test)
    f1 = f1_score(y_test, predictions, average='macro')
    # report = classification_report(y_test, predictions)
    # print(report)
    time_s = time.perf_counter() - start
    print(f'logregression ({penalty}) f1: {f1:.4f}, time (s): {time_s:.2f}')


def knearest(n_neighbors):
    start = time.perf_counter()

    X_train, x_test, Y_train, y_test = load_sets()

    reg_knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    reg_knn.fit(X_train, Y_train)
    predictions = reg_knn.predict(x_test)
    f1 = f1_score(y_test, predictions, average='macro')
    time_s = time.perf_counter() - start
    print(f'k-nearest neighbors ({n_neighbors}) f1: {f1:.4f}, time (s): {time_s:.2f}')


def adaboost(n):
    start = time.perf_counter()

    X_train, x_test, Y_train, y_test = load_sets()

    ada = AdaBoostClassifier(n_estimators=n, random_state=42)
    ada.fit(X_train, Y_train)
    predictions = ada.predict(x_test)
    f1 = f1_score(y_test, predictions, average='macro')
    time_s = time.perf_counter() - start
    print(f'adaboost ({n}) f1: {f1:.4f}, time (s): {time_s:.2f}')


def sgboost(n):
    start = time.perf_counter()

    X_train, x_test, Y_train, y_test = load_sets()

    sgb = GradientBoostingClassifier(n_estimators=n, random_state=42)
    sgb.fit(X_train, Y_train)
    predictions = sgb.predict(x_test)
    f1 = f1_score(y_test, predictions, average='macro')
    time_s = time.perf_counter() - start
    print(f'stochastic grad boost ({n}) f1: {f1:.4f}, time (s): {time_s:.2f}')


def mlpmnn(layer):

    start = time.perf_counter()
    X_train, x_test, Y_train, y_test = load_sets()

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train) 
    x_test = scaler.transform(x_test)  

    mlp = MLPClassifier(solver='adam', alpha=1e-5, activation='relu', max_iter=5000, 
                        learning_rate = 'adaptive',
                         hidden_layer_sizes=layer, random_state=42)
    mlp.fit(X_train, Y_train)
    predictions = mlp.predict(x_test)
    f1 = f1_score(y_test, predictions, average='macro')
    time_s = time.perf_counter() - start
    print(f'mlp {layer} f1: {f1:.4f}, time (s): {time_s:.2f}')



def load_sets():
    data = pd.read_csv('outputs/data_processed.csv', index_col=0, header=0)
    data = data.iloc[:, 3:]

    data['gz2class'] = data['gz2class'].str.replace('spiral', '0')
    data['gz2class'] = data['gz2class'].str.replace('elliptical', '1')
    data['gz2class'] = data['gz2class'].apply(pd.to_numeric)

    data_train = data.drop('gz2class', axis=1)
    data_label = data['gz2class'].values.tolist()
    data_train = data_train.to_numpy()

    X_train, x_test, Y_train, y_test = train_test_split(data_train, data_label, 
        test_size=0.2, random_state=42)

    return X_train, x_test, Y_train, y_test

if __name__ == '__main__':
    pass
    


    # random forest
    maxdepth = list(product([1, 5, 10, 15, 20, 30, 50]))
    # pool = mp.Pool(mp.cpu_count() - 4)
    # check = pool.starmap(randomforest, maxdepth)
    # pool.close()
    # pool.join()
    # 228k objects
    # randomforest (1) f1: 0.7303, time (s): 9.78
    # randomforest (5) f1: 0.7805, time (s): 32.23
    # randomforest (10) f1: 0.8082, time (s): 54.23
    # randomforest (15) f1: 0.8178, time (s): 71.60
    # randomforest (20) f1: 0.8203, time (s): 82.51
    # randomforest (30) f1: 0.8205, time (s): 90.19
    # randomforest (50) f1: 0.8210, time (s): 89.98 
    # 100k objects
    # randomforest (1) f1: 0.7442, time (s): 4.10
    # randomforest (5) f1: 0.8044, time (s): 13.09
    # randomforest (10) f1: 0.8322, time (s): 22.30
    # randomforest (15) f1: 0.8392, time (s): 29.00
    # randomforest (20) f1: 0.8385, time (s): 33.42
    # randomforest (30) f1: 0.8398, time (s): 35.61
    # randomforest (50) f1: 0.8395, time (s): 35.63
    # 50k objects
    # randomforest (1) f1: 0.7498, time (s): 1.98
    # randomforest (5) f1: 0.8073, time (s): 6.12
    # randomforest (10) f1: 0.8343, time (s): 10.08
    # randomforest (15) f1: 0.8413, time (s): 13.53
    # randomforest (20) f1: 0.8417, time (s): 15.17
    # randomforest (30) f1: 0.8410, time (s): 15.89
    # randomforest (50) f1: 0.8390, time (s): 15.79
    # 25k objects


    # adaboost
    n_leaners = list(product([1, 5, 10, 20, 50, 100, 200, 500]))
    # pool = mp.Pool(mp.cpu_count() - 4)
    # check = pool.starmap(adaboost, n_leaners)
    # pool.close()
    # pool.join()
    # 228k objects
    # adaboost (1) f1: 0.7267, time (s): 2.72
    # adaboost (5) f1: 0.7431, time (s): 5.19
    # adaboost (10) f1: 0.7666, time (s): 8.46
    # adaboost (20) f1: 0.7786, time (s): 14.56
    # adaboost (50) f1: 0.7912, time (s): 33.04
    # adaboost (100) f1: 0.7988, time (s): 62.84
    # adaboost (200) f1: 0.8040, time (s): 121.54
    # adaboost (500) f1: 0.8078, time (s): 293.48
    # 100k objects
    # adaboost (1) f1: 0.7495, time (s): 1.19
    # adaboost (5) f1: 0.7688, time (s): 2.27
    # adaboost (10) f1: 0.7875, time (s): 3.50
    # adaboost (20) f1: 0.7967, time (s): 6.07
    # adaboost (50) f1: 0.8093, time (s): 13.52
    # adaboost (100) f1: 0.8157, time (s): 25.77
    # adaboost (200) f1: 0.8213, time (s): 49.33
    # adaboost (500) f1: 0.8257, time (s): 120.40
    # 50k objects
    # adaboost (1) f1: 0.7629, time (s): 0.58
    # adaboost (5) f1: 0.7896, time (s): 1.08
    # adaboost (10) f1: 0.7964, time (s): 1.72
    # adaboost (20) f1: 0.8077, time (s): 2.93
    # adaboost (50) f1: 0.8171, time (s): 6.52
    # adaboost (100) f1: 0.8245, time (s): 12.74
    # adaboost (200) f1: 0.8287, time (s): 24.59
    # adaboost (500) f1: 0.8366, time (s): 58.97
    # 25k objects



    # stochastic gradient boosting
    n_leaners = list(product([1, 5, 10, 20, 50, 100, 200, 500]))
    # pool = mp.Pool(mp.cpu_count() - 4)
    # check = pool.starmap(sgboost, n_leaners)
    # pool.close()
    # pool.join()
    # 228k objects
    # stochastic grad boost (1) f1: 0.3646, time (s): 3.81
    # stochastic grad boost (5) f1: 0.7671, time (s): 10.41
    # stochastic grad boost (10) f1: 0.7804, time (s): 18.69
    # stochastic grad boost (20) f1: 0.7893, time (s): 34.74
    # stochastic grad boost (50) f1: 0.8021, time (s): 83.73
    # stochastic grad boost (100) f1: 0.8107, time (s): 162.53
    # stochastic grad boost (200) f1: 0.8169, time (s): 318.45
    # stochastic grad boost (500) f1: 0.8218, time (s): 774.50
    # 100k objects
    # stochastic grad boost (1) f1: 0.4028, time (s): 1.62
    # stochastic grad boost (5) f1: 0.7002, time (s): 4.32
    # stochastic grad boost (10) f1: 0.7824, time (s): 7.72
    # stochastic grad boost (20) f1: 0.8074, time (s): 14.27
    # stochastic grad boost (50) f1: 0.8226, time (s): 33.76
    # stochastic grad boost (100) f1: 0.8293, time (s): 66.33
    # stochastic grad boost (200) f1: 0.8353, time (s): 128.69
    # stochastic grad boost (500) f1: 0.8404, time (s): 316.01
    # 50k objects
    # stochastic grad boost (1) f1: 0.4030, time (s): 0.80
    # stochastic grad boost (5) f1: 0.7111, time (s): 2.01
    # stochastic grad boost (10) f1: 0.7692, time (s): 3.63
    # stochastic grad boost (20) f1: 0.8066, time (s): 6.54
    # stochastic grad boost (50) f1: 0.8260, time (s): 16.19
    # stochastic grad boost (100) f1: 0.8360, time (s): 31.99
    # stochastic grad boost (200) f1: 0.8390, time (s): 62.66
    # stochastic grad boost (500) f1: 0.8454, time (s): 151.51
    # 25k objects


    # k-nearest neighbors
    n_neighbors = list(product([1, 5, 10, 20, 30, 50]))
    # pool = mp.Pool(mp.cpu_count() - 4)
    # check = pool.starmap(knearest, n_neighbors)
    # pool.close()
    # pool.join()
    # 228k objects
    # k-nearest neighbors (1) f1: 0.7060, time (s): 131.96
    # k-nearest neighbors (20) f1: 0.7735, time (s): 152.52
    # k-nearest neighbors (10) f1: 0.7656, time (s): 152.52
    # k-nearest neighbors (5) f1: 0.7571, time (s): 152.52
    # k-nearest neighbors (30) f1: 0.7747, time (s): 152.55
    # k-nearest neighbors (50) f1: 0.7757, time (s): 152.56
    # 100k objects
    # k-nearest neighbors (1) f1: 0.7303, time (s): 24.73
    # k-nearest neighbors (20) f1: 0.7919, time (s): 34.49
    # k-nearest neighbors (30) f1: 0.7942, time (s): 34.54
    # k-nearest neighbors (5) f1: 0.7791, time (s): 34.61
    # k-nearest neighbors (10) f1: 0.7860, time (s): 34.64
    # k-nearest neighbors (50) f1: 0.7944, time (s): 34.70
    # 50k objects
    # k-nearest neighbors (1) f1: 0.7249, time (s): 7.07
    # k-nearest neighbors (5) f1: 0.7803, time (s): 8.81
    # k-nearest neighbors (50) f1: 0.7956, time (s): 8.89
    # k-nearest neighbors (20) f1: 0.7930, time (s): 8.91
    # k-nearest neighbors (30) f1: 0.7952, time (s): 9.05
    # k-nearest neighbors (10) f1: 0.7852, time (s): 9.07
    # 25k objects




    # svms
    kernels = list(product(['linear', 'poly', 'rbf']))
    # pool = mp.Pool(mp.cpu_count() - 4)
    # check = pool.starmap(svms, kernels)
    # pool.close()
    # pool.join()
    # with normalization
    # svm (poly) f1: 0.7581795867231094
    # svm (linear) f1: 0.7416363435494805
    # svm (rbf) f1: 0.7340012637340211
    # without normalization
    # svm (linear) f1: 0.3645528726989167
    # svm (poly) f1: 0.3645528726989167
    # svm (rbf) f1: 0.3645528726989167





    # multi-layer perceptron
    layers = [(1,1), (2,2), (5,2), (5,5), (2,2,2), (4,4,4), 
            (8,8,8), (2,2,2,2), (5, 5, 5, 5), (8,8,8,8)]
    for layer in layers:
        mlpmnn(layer)

    # adam relu, 228k objects
    # mlp (1, 1) f1: 0.7853, time (s): 11.27
    # mlp (2, 2) f1: 0.7962, time (s): 12.48
    # mlp (5, 2) f1: 0.8118, time (s): 16.81
    # mlp (5, 5) f1: 0.8094, time (s): 15.97
    # mlp (2, 2, 2) f1: 0.7859, time (s): 33.54
    # mlp (4, 4, 4) f1: 0.8099, time (s): 19.06
    # mlp (8, 8, 8) f1: 0.8167, time (s): 17.56
    # mlp (2, 2, 2, 2) f1: 0.7976, time (s): 24.31
    # mlp (5, 5, 5, 5) f1: 0.8100, time (s): 32.56
    # mlp (8, 8, 8, 8) f1: 0.8154, time (s): 23.34
    # 100k objects
    # mlp (1, 1) f1: 0.8078, time (s): 8.70
    # mlp (2, 2) f1: 0.8167, time (s): 8.14
    # mlp (5, 2) f1: 0.8182, time (s): 8.96
    # mlp (5, 5) f1: 0.8310, time (s): 12.60
    # mlp (2, 2, 2) f1: 0.8082, time (s): 15.14
    # mlp (4, 4, 4) f1: 0.8318, time (s): 11.11
    # mlp (8, 8, 8) f1: 0.8365, time (s): 13.78
    # mlp (2, 2, 2, 2) f1: 0.8142, time (s): 5.42
    # mlp (5, 5, 5, 5) f1: 0.8315, time (s): 11.69
    # mlp (8, 8, 8, 8) f1: 0.8348, time (s): 14.87
    # 50k objects
    # mlp (1, 1) f1: 0.8141, time (s): 6.42
    # mlp (2, 2) f1: 0.8145, time (s): 7.04
    # mlp (5, 2) f1: 0.8267, time (s): 6.69
    # mlp (5, 5) f1: 0.8408, time (s): 4.58
    # mlp (2, 2, 2) f1: 0.8133, time (s): 10.26
    # mlp (4, 4, 4) f1: 0.8380, time (s): 6.43
    # mlp (8, 8, 8) f1: 0.8414, time (s): 6.50
    # mlp (2, 2, 2, 2) f1: 0.8178, time (s): 2.37
    # mlp (5, 5, 5, 5) f1: 0.8396, time (s): 6.06
    # mlp (8, 8, 8, 8) f1: 0.8417, time (s): 9.04
    # 25k objects
    # mlp (1, 1) f1: 0.8217, time (s): 3.20
    # mlp (2, 2) f1: 0.8143, time (s): 2.04
    # mlp (5, 2) f1: 0.8340, time (s): 2.52
    # mlp (5, 5) f1: 0.8545, time (s): 4.12
    # mlp (2, 2, 2) f1: 0.8169, time (s): 3.34
    # mlp (4, 4, 4) f1: 0.8386, time (s): 3.77
    # mlp (8, 8, 8) f1: 0.8532, time (s): 3.49
    # mlp (2, 2, 2, 2) f1: 0.8298, time (s): 1.94
    # mlp (5, 5, 5, 5) f1: 0.8559, time (s): 3.47
    # mlp (8, 8, 8, 8) f1: 0.8557, time (s): 3.29