# directory tree : root - platform - architecture - packer - option - binary
# input root_directory
# circulate every sub directory, and extract data from binary

import pefile
import os
import csv
import Queue
import threading
from datetime import datetime
from tqdm import tqdm

# base variables
root_file = "..\\binary"
output_directory = ".\\data\\"
output_time = datetime.today().strftime("%Y%m%d%H%M%S")
output_file_name = 'byte_feature'
# commonly used variables
output_file = output_directory + output_file_name + output_time + '.csv'
# shared variables for multi-thread
file_list_queue = Queue.Queue()
thread_lock = threading.Lock()


class Extractor(threading.Thread):
    def run(self):
        while file_list_queue.qsize() > 0:
            path = file_list_queue.get()

            try:
                byte_sequences = extract_byte_from_binary(path)
            except:
                continue

            write_byte_information(path, byte_sequences)


def print_time():
    print datetime.today()


def is_directory(path):
    return os.path.isdir(path)


def is_file(path):
    return os.path.isfile(path)


def get_file_list(path):
    return os.listdir(path)


def write_byte_information(path, byte_sequences):
    output = open(output_file, 'a')

    dir_names = os.path.dirname(path).split('\\')
    byte_sequences = map(str, byte_sequences)
    name = os.path.basename(path)
    option = dir_names[-1]
    packer = dir_names[-2]
    architecture = dir_names[-3]
    platform = dir_names[-4]
    data = ','.join([platform, architecture, packer, option, name]) + ',' + ','.join(byte_sequences) + '\n'

    thread_lock.acquire()
    output.writelines(data)
    thread_lock.release()

    output.close()


def extract_byte_from_binary(path):
    pe = pefile.PE(path)
    seqs = pe.get_data(pe.OPTIONAL_HEADER.AddressOfEntryPoint, 30).encode('hex')
    split_seq = []
    for i in range(0, 30, 2):
        # encode, case by case
        # split_seq.append(seqs[i:i + 2])
        split_seq.append(int(seqs[i:i + 2], 16))
    return split_seq


def set_queue(path):
    file_list = None
    if is_directory(path):
        file_list = get_file_list(path)
        path = '\\'.join([path, ''])
    else:
        file_list_queue.put(path)

    if file_list is not None:
        for f in file_list:
            set_queue(path + f)


def extract():
    def create_thread(n):
        thread_array = []
        for i in range(n):
            thread_array.append(Extractor())
        return thread_array

    def start_thread(threads):
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    print_time()
    with open(output_file, 'wb') as output:
        column = ['platform', 'architecture', 'packer', 'option', 'name', 'seq1', 'seq2', 'seq3', 'seq4', 'seq5', 'seq6', 'seq7', 'seq8', 'seq9', 'seq10', 'seq11', 'seq12', 'seq13', 'seq14', 'seq15']
        output.writelines(','.join(column) + '\n')
    set_queue(root_file)
    print 'Number of files : ', file_list_queue.qsize()
    threads = create_thread(3)
    start_thread(threads)
    print_time()

extract()
