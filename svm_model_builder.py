from sklearn import svm
from sklearn.externals import joblib
from datetime import datetime
import time
import os
import re

model_directory = './model/'


def save_svm_model(clf, pkl_name):
    model_path = model_directory + '{}.pkl'.format(pkl_name)
    joblib.dump(clf, model_path)


def make_model_directory():
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)


def modeling(training_data, kernel, c, gamma):
    feature = training_data.iloc[:, 5:20]
    cls = training_data.iloc[:, 1:3]

    for i in range(len(cls)):
        cls.iloc[i, ] = ''.join(cls.iloc[i, ])
    # print cls.iloc[:, 0]

    clf = svm.SVC(kernel=kernel, C=c, gamma=gamma)
    if type(kernel) is str:
        clf = svm.SVC(kernel=kernel)
    print '{}({} {}) training start time {}'.format(kernel, c, gamma, datetime.today())
    start = time.time()
    clf.fit(feature, cls.iloc[:, 0])
    training_time = time.time() - start
    # training_complete_time = datetime.today().strftime("%Y%m%d%H%M%S")
    training_complete_time = datetime.today().strftime("%m%d%H")
    print '{}({} {}) training complete time {}'.format(kernel, c, gamma, datetime.today())

    kernel_name = str(kernel)
    pattern = re.compile('<.+>')
    if pattern.match(kernel_name):
        kernel_name = kernel_name.split(' ')[1]
    pkl_name = '{}_{}_{}_{}_{}'.format(kernel_name, c, gamma, training_time, training_complete_time)
    save_svm_model(clf, pkl_name)
