from data_processor import *
from svm_model_builder import *
from datetime import datetime
from tqdm import tqdm
import pandas
import csv
import time
import os
import re

# # global variables
# path variables
# date = datetime.today().strftime("%Y%m%d%H%M%S")
# date = datetime.today().strftime("%Y%m%d")
report_base_directory = './report/'
# file write variables
architecture_list = ['32bit', '64bit']
packer_list = ['ASPack', 'ASProtect', 'EnigmaProtector', 'mpress', 'Obsidium', 'Original', 'PESpin', 'Themida', 'UPX', 'VMProtect']
mis_classification_row = ['platform', 'architecture', 'packer', 'option', 'name', 'kernel', 'predict']


def make_directory(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def make_report_base_directory():
    if not os.path.isdir(report_base_directory):
        os.mkdir(report_base_directory)


def load_model(path):
    svm_model = joblib.load(path)
    return svm_model


def classify(clf, test_data):
    def _classify(_clf, _test_data):
        _test_feature = _test_data.iloc[:, 5:20]
        # test_class = test_data.iloc[:, 11]
        _count = 0.0
        kernel = _clf.kernel
        pattern = re.compile('<.+>')
        if pattern.match(str(kernel)):
            kernel = str(kernel).split(' ')[1]
        _classified_class = [['predict']]
        _mis_classification_datas = []
        for j in tqdm([i for i in range(len(_test_feature))], desc='{}'.format(kernel)):
            predicted_class = _clf.predict(_test_feature[j:j + 1])
            _classified_class.append(predicted_class[0])
            origin_class = ''.join(_test_data.iloc[j, 1:3])
            if origin_class == predicted_class[0]:
                _count += 1
            else:
                mis_classification_data = [_test_data.iloc[j, 0], _test_data.iloc[j, 1], _test_data.iloc[j, 2],
                                           _test_data.iloc[j, 3], _test_data.iloc[j, 4], kernel, predicted_class[0]]
                # mis_feature = _test_data.iloc[j, 0:5].tolist()
                _mis_classification_datas.append(mis_classification_data)

        _mis_classification_datas = pandas.DataFrame(_mis_classification_datas, columns=mis_classification_row)
        _classified_class = pandas.DataFrame(_classified_class[1:], columns=_classified_class[0])
        return _classified_class, _count, _mis_classification_datas

    def analyze_performance_measure(_classified_class, _test_data):
        def count_element(_architecture, _packer, _test_data, _classified_class):
            clazz = ''.join([_architecture, _packer])
            _tp, _fp, _fn = 0, 0, 0
            _test_feature = _test_data.iloc[:, 5:20]
            for i in range(len(_test_feature)):
                o_class = ''.join(_test_data.iloc[i, 1:3])
                p_class = _classified_class.iloc[i, 0]
                if clazz == o_class and o_class == p_class:
                    _tp += 1
                elif clazz != o_class and clazz == p_class:
                    _fp += 1
                elif clazz == o_class and o_class != p_class:
                    _fn += 1
            return _fn, _fp, _tp

        def calculate_measure(_fn, _fp, _tp):
            _precision, _recall, _f1_measure = -1, -1, -1
            try:
                _precision = float(_tp) / (_tp + _fp)
                _recall = float(_tp) / (_tp + _fn)
                _f1_measure = 2 * _precision * _recall / (_precision + _recall)
            except ZeroDivisionError:
                pass
            return _f1_measure, _precision, _recall

        _precision_data, _recall_data, _f_measure_data = [], [], []
        for architecture in architecture_list:
            for packer in packer_list:
                fn, fp, tp = count_element(architecture, packer, _test_data, _classified_class)
                f1_measure, precision, recall = calculate_measure(fn, fp, tp)
                _precision_data.append(precision)
                _recall_data.append(recall)
                _f_measure_data.append(f1_measure)
        return _f_measure_data, _precision_data, _recall_data

    classified_class, count, mis_classification_datas = _classify(clf, test_data)
    f_measure_data, precision_data, recall_data = analyze_performance_measure(classified_class, test_data)
    test_feature = test_data.iloc[:, 5:20]
    accuracy = count / len(test_feature)

    return accuracy, precision_data, recall_data, f_measure_data, mis_classification_datas


def generate_f_measure_data_columns(test_data):
    first_column = []
    second_column = []
    for architecture in architecture_list:
        for packer in packer_list:
            cls = ''.join([architecture, packer])
            cls_count = len(test_data[(test_data.architecture == architecture) & (test_data.packer == packer)])
            first_column.append(cls)
            second_column.append(cls_count)
    return first_column, second_column


def run_classification(model_path, test_set):
    clf = load_model(model_path)
    c, gamma = clf.get_params()['C'], clf.get_params()['gamma']
    kernel_name = str(clf.kernel)
    pattern = re.compile('<.+>')
    if pattern.match(kernel_name):
        kernel_name = kernel_name.split(' ')[1]
    date = datetime.today().strftime("%Y%m%d%H%M%S")
    model_report_directory = '{}{}_{}_{}_{}/'.format(report_base_directory, kernel_name, c, gamma, date)
    make_directory(model_report_directory)

    svm_report = open(model_report_directory + 'accuracy.csv', 'wb')
    svm_writer = csv.writer(svm_report, delimiter=',')
    svm_writer.writerow(['kernel', 'C', 'gamma', 'accuracy', 'execution_time'])

    mis_classification_data_frame = pandas.DataFrame([mis_classification_row], columns=mis_classification_row)

    measure_data = pandas.DataFrame()
    first, second = generate_f_measure_data_columns(test_set)
    measure_data.insert(0, 'Class', first)
    measure_data.insert(1, '#', second)

    start = time.time()
    accuracy, precision, recall, f_measure, mis_classification_datas = classify(clf, test_set)
    mis_classification_data_frame = mis_classification_data_frame.append(mis_classification_datas)
    execution_time = time.time() - start

    measure_data.insert(2, 'precision', precision)
    measure_data.insert(3, 'recall', recall)
    measure_data.insert(4, 'f-measure', f_measure)
    svm_writer.writerow([kernel_name, c, gamma, accuracy, execution_time])

    mis_classification_data_frame.to_csv(model_report_directory + 'mis_classified_data.csv', index=False, header=False)
    measure_data.to_csv(model_report_directory + 'f_measure.csv', index=False)
    svm_report.close()
