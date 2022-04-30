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

    X_train, x_test, Y_train, y_test = load_sets()
    
    # initialize model
    clf = RandomForestClassifier(max_depth=max_depth, random_state=42)

    # train the model
    clf = clf.fit(X_train, Y_train)

    # make predictions using the same features
    predictions = clf.predict(x_test)
    f1 = f1_score(y_test, predictions, average='macro')
    print(f'randomforest ({max_depth}) f1: {f1:.4f}')


def svms(kernel):

    X_train, x_test, Y_train, y_test = load_sets()

    # initialize model
    svc = svm.SVC(kernel=kernel)
    # train the model
    svc = svc.fit(X_train, Y_train)
    # make predictions using the same features
    predictions = svc.predict(x_test)
    f1 = f1_score(y_test, predictions, average='macro')
    print(f'svm ({kernel}) f1: {f1:.4f}')


def logregression(penalty):

    X_train, x_test, Y_train, y_test = load_sets()

    logreg = LogisticRegression(penalty=penalty)
    logreg.fit(X_train, Y_train)
    predictions = logreg.predict(x_test)
    f1 = f1_score(y_test, predictions, average='macro')
    print(f'logregression ({penalty}) f1: {f1:.4f}')
    # report = classification_report(y_test, predictions)
    # print(report)


def knearest(n_neighbors):

    X_train, x_test, Y_train, y_test = load_sets()

    reg_knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    reg_knn.fit(X_train, Y_train)
    predictions = reg_knn.predict(x_test)
    f1 = f1_score(y_test, predictions, average='macro')
    print(f'k-nearest neighbors ({n_neighbors}) f1: {f1:.4f}')


def adaboost(n):

    X_train, x_test, Y_train, y_test = load_sets()

    ada = AdaBoostClassifier(n_estimators=n, random_state=42)
    ada.fit(X_train, Y_train)
    predictions = ada.predict(x_test)
    f1 = f1_score(y_test, predictions, average='macro')
    print(f'adaboost ({n}) f1: {f1:.4f}')


def sgboost(n):

    X_train, x_test, Y_train, y_test = load_sets()

    sgb = GradientBoostingClassifier(n_estimators=n, random_state=42)
    sgb.fit(X_train, Y_train)
    predictions = sgb.predict(x_test)
    f1 = f1_score(y_test, predictions, average='macro')
    print(f'stochastic grad boost ({n}) f1: {f1:.4f}')


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
    print(f'multi-layer perceptron {layer} f1: {f1:.4f}, time (s): {time.perf_counter() - start}')



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
    pool = mp.Pool(mp.cpu_count() - 4)
    
    # check = pool.starmap(randomforest, maxdepth)
    # pool.close()
    # pool.join()
    # without normalization
    # randomforest (1) f1: 0.7303454914593727
    # randomforest (5) f1: 0.7804921516234901
    # randomforest (10) f1: 0.8081521435845178
    # randomforest (15) f1: 0.8177741515693964
    # randomforest (20) f1: 0.8202972259259638
    # randomforest (30) f1: 0.8204639999130131
    # randomforest (50) f1: 0.8209898238067115
    # 50k objects
    # randomforest (1) f1: 0.7760615241070277
    # randomforest (5) f1: 0.8252447262701554
    # randomforest (10) f1: 0.8505456092818902
    # randomforest (15) f1: 0.8571269929828806
    # randomforest (20) f1: 0.8598315759178976
    # randomforest (30) f1: 0.85631564578933
    # randomforest (50) f1: 0.8548788022472233

    # adaboost
    n_leaners = list(product([1, 5, 10, 20, 50, 100, 200, 500]))
    check = pool.starmap(adaboost, n_leaners)
    pool.close()
    pool.join()
    # adaboost (1) f1: 0.7267489978996022
    # adaboost (5) f1: 0.7431069654743305
    # adaboost (10) f1: 0.7665777833409222
    # adaboost (20) f1: 0.7786024321322966
    # adaboost (50) f1: 0.79121652923077
    # adaboost (100) f1: 0.7988385960074693
    # adaboost (200) f1: 0.804033147046551
    # adaboost (500) f1: 0.8078148310623934
    # 50k objects
    # adaboost (1) f1: 0.7870536316509607
    # adaboost (5) f1: 0.7944092582153397
    # adaboost (10) f1: 0.8044442515411168
    # adaboost (20) f1: 0.8152179135145923
    # adaboost (50) f1: 0.8370308477597781
    # adaboost (100) f1: 0.8458370466081858
    # adaboost (200) f1: 0.8476876502472999
    # adaboost (500) f1: 0.8507049901462904

    # stochastic gradient boosting
    n_leaners = list(product([1, 5, 10, 20, 50, 100, 200, 500]))
    # check = pool.starmap(sgboost, n_leaners)
    # pool.close()
    # pool.join()
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

    # multi-layer perceptron
    layers = [(1,1), (2,2), (5,2), (5,5), (2,2,2), (4,4,4), (8,8,8), (2,2,2,2), (5, 5, 5, 5), (8,8,8,8)]
    # for layer in layers:
    #     mlpmnn(layer)
    # check = pool.starmap(mlpmnn, layers)
    # pool.close()
    # pool.join()

    # adam relu, 228k objects
    # multi-layer perceptron (1, 1) f1: 0.7853407542214266, time (s): 11.7059435
    # multi-layer perceptron (2, 2) f1: 0.7962184542367859, time (s): 12.386172834
    # multi-layer perceptron (5, 2) f1: 0.811794594662048, time (s): 16.756201417
    # multi-layer perceptron (5, 5) f1: 0.8093591644090954, time (s): 16.007910124999995
    # multi-layer perceptron (2, 2, 2) f1: 0.7858913183339118, time (s): 33.928735333000006
    # multi-layer perceptron (4, 4, 4) f1: 0.8099418323374714, time (s): 19.162512958000008
    # multi-layer perceptron (8, 8, 8) f1: 0.8166629714689388, time (s): 17.631497666000016
    # multi-layer perceptron (2, 2, 2, 2) f1: 0.7976289754713084, time (s): 24.893360290999993
    # multi-layer perceptron (5, 5, 5, 5) f1: 0.8100497838476945, time (s): 33.158824041
    # multi-layer perceptron (8, 8, 8, 8) f1: 0.815352430461644, time (s): 23.542501458999993
    
    # adam relu, 150k objects
    # multi-layer perceptron (1, 1) f1: 0.8036443661227938, time (s): 10.670384417000001
    # multi-layer perceptron (2, 2) f1: 0.8093040293040292, time (s): 9.408379833000001
    # multi-layer perceptron (5, 2) f1: 0.8102257662518857, time (s): 11.561299207999998
    # multi-layer perceptron (5, 5) f1: 0.8223366115164215, time (s): 18.365282959000005
    # multi-layer perceptron (2, 2, 2) f1: 0.8039292567198577, time (s): 21.283074625000005
    # multi-layer perceptron (4, 4, 4) f1: 0.8196108432820034, time (s): 16.74146574999999
    # multi-layer perceptron (8, 8, 8) f1: 0.8221153383262403, time (s): 14.019208374999991
    # multi-layer perceptron (2, 2, 2, 2) f1: 0.8077052506081321, time (s): 21.78746975
    # multi-layer perceptron (5, 5, 5, 5) f1: 0.8195925319299953, time (s): 12.43264145800002
    # multi-layer perceptron (8, 8, 8, 8) f1: 0.8251028731973047, time (s): 16.592592584000016

    # adam relu, 100k objects
    # multi-layer perceptron (1, 1) f1: 0.8077848907689893, time (s): 9.047992958
    # multi-layer perceptron (2, 2) f1: 0.8167027858972844, time (s): 8.157504416
    # multi-layer perceptron (5, 2) f1: 0.8182414019443605, time (s): 8.928623125000001
    # multi-layer perceptron (5, 5) f1: 0.8309747803501845, time (s): 12.778056375000002
    # multi-layer perceptron (2, 2, 2) f1: 0.8081939897332482, time (s): 15.139135332999999
    # multi-layer perceptron (4, 4, 4) f1: 0.8318014705882353, time (s): 11.028988249999998
    # multi-layer perceptron (8, 8, 8) f1: 0.8364666193789837, time (s): 13.763977874999995
    # multi-layer perceptron (2, 2, 2, 2) f1: 0.8142163301063174, time (s): 5.403228999999996
    # multi-layer perceptron (5, 5, 5, 5) f1: 0.8315257198930507, time (s): 11.609597707999995
    # multi-layer perceptron (8, 8, 8, 8) f1: 0.8348342937790401, time (s): 14.594578041999995

    # 50k objects
    # multi-layer perceptron (1, 1) f1: 0.8140772437471969, time (s): 7.274933334
    # multi-layer perceptron (2, 2) f1: 0.814509848550778, time (s): 7.437800667000001
    # multi-layer perceptron (5, 2) f1: 0.8267244604380629, time (s): 6.965831000000001
    # multi-layer perceptron (5, 5) f1: 0.840847701610899, time (s): 4.805234500000001
    # multi-layer perceptron (2, 2, 2) f1: 0.8133321504009798, time (s): 10.722146167000002
    # multi-layer perceptron (4, 4, 4) f1: 0.838006836111516, time (s): 6.6532272500000005
    # multi-layer perceptron (8, 8, 8) f1: 0.8413785220683463, time (s): 6.709666374999998
    # multi-layer perceptron (2, 2, 2, 2) f1: 0.8178480918306501, time (s): 2.4366880840000036
    # multi-layer perceptron (5, 5, 5, 5) f1: 0.8396082163364378, time (s): 6.297046999999999
    # multi-layer perceptron (8, 8, 8, 8) f1: 0.841715495029517, time (s): 9.361346333

    # 25k objects
    # multi-layer perceptron (1, 1) f1: 0.8216925746077572, time (s): 3.5875596660000006
    # multi-layer perceptron (2, 2) f1: 0.8143106676947771, time (s): 2.070206583
    # multi-layer perceptron (5, 2) f1: 0.8340285477682073, time (s): 2.5235157499999996
    # multi-layer perceptron (5, 5) f1: 0.8545441926568533, time (s): 4.124674875
    # multi-layer perceptron (2, 2, 2) f1: 0.816936776936591, time (s): 3.300061166999999
    # multi-layer perceptron (4, 4, 4) f1: 0.8385536238884084, time (s): 3.714399708000002
    # multi-layer perceptron (8, 8, 8) f1: 0.8532134418606854, time (s): 3.4175369589999995
    # multi-layer perceptron (2, 2, 2, 2) f1: 0.829767134368895, time (s): 1.9169627499999997
    # multi-layer perceptron (5, 5, 5, 5) f1: 0.8559468641053892, time (s): 3.4315830829999996
    # multi-layer perceptron (8, 8, 8, 8) f1: 0.8556930463084556, time (s): 3.2679395830000004

    