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


def randomforest(max_depth):

    X_train, x_test, Y_train, y_test = load_sets()
    
    # initialize model
    clf = RandomForestClassifier(max_depth=max_depth, random_state=42)

    # train the model
    clf = clf.fit(X_train, Y_train)

    # make predictions using the same features
    predictions = clf.predict(x_test)
    f1 = f1_score(y_test, predictions, average='macro')
    print(f'randomforest ({max_depth}) f1: {f1}')


def svms(kernel):

    X_train, x_test, Y_train, y_test = load_sets()

    # initialize model
    svc = svm.SVC(kernel=kernel)
    # train the model
    svc = svc.fit(X_train, Y_train)
    # make predictions using the same features
    predictions = svc.predict(x_test)
    f1 = f1_score(y_test, predictions, average='macro')
    print(f'svm ({kernel}) f1: {f1}')


def logregression(penalty):

    X_train, x_test, Y_train, y_test = load_sets()

    logreg = LogisticRegression(penalty=penalty)
    logreg.fit(X_train, Y_train)
    predictions = logreg.predict(x_test)
    f1 = f1_score(y_test, predictions, average='macro')
    print(f'logregression ({penalty}) f1: {f1}')
    # report = classification_report(y_test, predictions)
    # print(report)


def knearest(n_neighbors):

    X_train, x_test, Y_train, y_test = load_sets()

    reg_knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    reg_knn.fit(X_train, Y_train)
    predictions = reg_knn.predict(x_test)
    f1 = f1_score(y_test, predictions, average='macro')
    print(f'k-nearest neighbors ({n_neighbors}) f1: {f1}')


def adaboost(n):

    X_train, x_test, Y_train, y_test = load_sets()

    ada = AdaBoostClassifier(n_estimators=n, random_state=42)
    ada.fit(X_train, Y_train)
    predictions = ada.predict(x_test)
    f1 = f1_score(y_test, predictions, average='macro')
    print(f'adaboost ({n}) f1: {f1}')


def sgboost(n):

    X_train, x_test, Y_train, y_test = load_sets()

    sgb = GradientBoostingClassifier(n_estimators=n, random_state=42)
    sgb.fit(X_train, Y_train)
    predictions = sgb.predict(x_test)
    f1 = f1_score(y_test, predictions, average='macro')
    print(f'stochastic grad boost ({n}) f1: {f1}')



def load_sets():
    data = pd.read_csv('outputs/data_processed.csv', index_col=0, header=0)
    data = data.iloc[:, 3:]

    data['gz2class'] = data['gz2class'].str.replace('spiral', '0')
    data['gz2class'] = data['gz2class'].str.replace('elliptical', '1')
    data['gz2class'] = data['gz2class'].apply(pd.to_numeric)

    data_train = data.drop('gz2class', axis=1)
    data_label = data['gz2class'].values.tolist()

    X_train, x_test, Y_train, y_test = train_test_split(data_train, data_label, 
        test_size=0.2, random_state=42)

    return X_train, x_test, Y_train, y_test

if __name__ == '__main__':
    pass
    


    # random forest
    maxdepth = list(product([1, 5, 10, 15, 20, 30, 50]))
    pool = mp.Pool(mp.cpu_count() - 2)
    # check = pool.starmap(randomforest, maxdepth)
    # pool.close()
    # pool.join()
    # with normalization
    # randomforest (1) f1: 0.7246356463185823
    # randomforest (5) f1: 0.7766658832853917
    # randomforest (10) f1: 0.8043900748823363
    # randomforest (15) f1: 0.8162582814260413
    # randomforest (20) f1: 0.8181926399412605
    # randomforest (30) f1: 0.8195233000815982
    # randomforest (50) f1: 0.8195306792494652
    # without normalization
    # randomforest (1) f1: 0.7303454914593727
    # randomforest (5) f1: 0.7804921516234901
    # randomforest (10) f1: 0.8081521435845178
    # randomforest (15) f1: 0.8177741515693964
    # randomforest (20) f1: 0.8202972259259638
    # randomforest (30) f1: 0.8204639999130131
    # randomforest (50) f1: 0.8209898238067115

    # adaboost
    n_leaners = list(product([1, 5, 10, 20, 50, 100, 200, 500]))
    # check = pool.starmap(adaboost, n_leaners)
    # pool.close()
    # pool.join()
    # adaboost (1) f1: 0.7267489978996022
    # adaboost (5) f1: 0.7431069654743305
    # adaboost (10) f1: 0.7665777833409222
    # adaboost (20) f1: 0.7786024321322966
    # adaboost (50) f1: 0.79121652923077
    # adaboost (100) f1: 0.7988385960074693
    # adaboost (200) f1: 0.804033147046551
    # adaboost (500) f1: 0.8078148310623934

    # stochastic gradient boosting
    n_leaners = list(product([1, 5, 10, 20, 50, 100, 200, 500]))
    check = pool.starmap(sgboost, n_leaners)
    pool.close()
    pool.join()
    # stochastic grad boost (1) f1: 0.3645528726989167
    # stochastic grad boost (5) f1: 0.7671040215433715
    # stochastic grad boost (10) f1: 0.7804324298444166
    # stochastic grad boost (20) f1: 0.7892545080020885
    # stochastic grad boost (50) f1: 0.8021472198380418
    # stochastic grad boost (100) f1: 0.8106613365161859
    # stochastic grad boost (200) f1: 0.8168589265684792
    # stochastic grad boost (500) f1: 0.8217757750515524
    
    # svms
    kernels = list(product(['linear', 'poly', 'rbf']))
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

    # logistic regression
    penalties = ['l2']
    # for pen in penalties:
    #     logregression(pen)
    # with normalization
    # logregression (l2) f1: 0.7331682730530195
    # without normalization
    # logregression (l2) f1: 0.7330428999395542

    # k-nearest neighbors
    n_neighbors = list(product([1, 5, 10, 20, 30, 50]))
    # check = pool.starmap(knearest, n_neighbors)
    # pool.close()
    # pool.join()
    # k-nearest neighbors (1) f1: 0.6957263451300708
    # k-nearest neighbors (5) f1: 0.7401451419299696
    # k-nearest neighbors (10) f1: 0.7505500016631921
    # k-nearest neighbors (20) f1: 0.7607139054678177
    # k-nearest neighbors (30) f1: 0.7614413598111676
    # k-nearest neighbors (50) f1: 0.7618164237789349
    # without normalization
    # k-nearest neighbors (1) f1: 0.7059521778693815
    # k-nearest neighbors (5) f1: 0.7571271414602448
    # k-nearest neighbors (10) f1: 0.7656058554900411
    # k-nearest neighbors (20) f1: 0.7735366468960683
    # k-nearest neighbors (30) f1: 0.7746545668068594
    # k-nearest neighbors (50) f1: 0.7757132046270314

