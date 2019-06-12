from sklearn.utils import shuffle
import pandas

feature_data_path = "./data/byte_feature20171114182226.csv"
sample_data_path = './data/sample_data.csv'
training_data_path = './data/training_sample_data.csv'
test_data_path = './data/test_sample_data.csv'

# features = [platform, architecture, packer, option, name, seq1, seq2, seq3 ... seq15]
header = ['platform', 'architecture', 'packer', 'option', 'name', 'seq1', 'seq2', 'seq3', 'seq4', 'seq5', 'seq6', 'seq7', 'seq8', 'seq9', 'seq10', 'seq11', 'seq12', 'seq13', 'seq14', 'seq15']
architecture_list = ['32bit', '64bit']
packer_list = ['ASPack', 'ASProtect', 'EnigmaProtector', 'mpress', 'Themida', 'Original', 'Obsidium', 'PESpin', 'UPX', 'VMProtect']


def save_data_to_csv(path, data):
    """
    :param path: string
    :param data: dataFrame
    :return:
    """
    # with open(path, 'w') as f:
    #     pickle.dump(data, f)
    data.columns = header
    data.to_csv(path, index=False, header=True)


# date structure ==> [platform architecture packer option name features*]
def load_csv_data(file_path):
    data = pandas.read_csv(file_path)
    return data


def divide_data(data, number_of_training):
    training_data_frame = pandas.DataFrame(columns=header)
    test_data_frame = pandas.DataFrame(columns=header)

    total = data.name.count()
    for architecture in architecture_list:
        data_filtered_architecture = data[data.architecture == architecture]
        for packer in packer_list:
            data_filtered_architecture_packer = data_filtered_architecture[data_filtered_architecture.packer == packer]
            length = data_filtered_architecture_packer.name.count()
            training_data_frame = training_data_frame.append(data_filtered_architecture_packer[:length * number_of_training / total])
            test_data_frame = test_data_frame.append(data_filtered_architecture_packer[length * number_of_training / total:])

    save_data_to_csv(training_data_path, training_data_frame)
    save_data_to_csv(test_data_path, test_data_frame)


def sampling(data, number_of_sample):
    sampling_data_frame = pandas.DataFrame(columns=header)

    data = shuffle(shuffle(data))
    total = data.name.count()
    for architecture in architecture_list:
        data_filtered_architecture = data[data.architecture == architecture]
        for packer in packer_list:
            data_filtered_architecture_packer = data_filtered_architecture[data_filtered_architecture.packer == packer]
            length = data_filtered_architecture_packer.name.count()
            sampling_data_frame = sampling_data_frame.append(data_filtered_architecture_packer[:length*number_of_sample/total])
    sampling_data_frame = sampling_data_frame.append(data[:(number_of_sample-sampling_data_frame.name.count())])
    save_data_to_csv(sample_data_path, sampling_data_frame)


# test function
def count_class(data):
    count_of_class = 0
    for architecture in architecture_list:
        for packer in packer_list:
            filtered_data = data[(data.architecture == architecture) & (data.packer == packer)]
            if filtered_data.name.count() > 0:
                count_of_class += 1
    print count_of_class
