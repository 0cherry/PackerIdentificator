from data_processor import *
from svm_model_builder import *
from svm_classifier import *
from kernel_function import *
from functools import partial
import math
import multiprocessing as mp

# kernel_list = ['rbf', rbf_lcs_kernel, rbf_ed_kernel, rbf_ngram_kernel, rbf_nh_kernel]
kernel_list = [rbf_ngram_kernel, rbf_nh_kernel]
Cs = [math.pow(2, i) for i in range(2, 7)]
gammas = [math.pow(2, i) for i in range(-7, -2)]


def parallel_supported_modeling(args):
    d, k, c, g = args[0], args[1], args[2], args[3]
    modeling(d, k, c, g)


# date structure ==> [platform architecture packer option name features*]
def get_model_list(path):
    model_list = []
    for model in os.listdir(path):
        model_full_path = ''.join([model_directory, model])
        model_list.append(model_full_path)
    return model_list

if __name__ == '__main__':
    ### divide data (training / test)
    feature_data = load_csv_data(feature_data_path)
    number_of_training = 10000
    divide_data(feature_data, number_of_training)

    ### modeling
    training_data = load_csv_data(training_data_path)
    make_model_directory()

    work_parameter = []

    # Fill work_queue
    for c in Cs:
        for gamma in gammas:
            for kernel in kernel_list:
                work_parameter.append((training_data, kernel, c, gamma))

    pool = mp.Pool(processes=(mp.cpu_count()-1))
    pool.map(parallel_supported_modeling, work_parameter)
    print "modeling is completed"

    ### classifying
    test_data = load_csv_data(test_data_path)
    make_report_base_directory()

    # Fill work_model_path
    work_model_path = get_model_list('./model')

    pool = mp.Pool(processes=(mp.cpu_count()-1))
    pool.map(partial(run_classification, test_set=test_data), work_model_path)
    print "classification is completed"
