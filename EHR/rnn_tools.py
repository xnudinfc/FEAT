import rootpath

rootpath.append()
import pickle
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import copy


def load_data(training_file, validation_file, testing_file):
    train = np.array(pickle.load(open(training_file, 'rb')))
    validate = np.array(pickle.load(open(validation_file, 'rb')))
    test = np.array(pickle.load(open(testing_file, 'rb')))
    return train, validate, test


def pad_matrix(seq_diagnosis_codes, seq_labels, options):
    lengths = np.array([len(seq) for seq in seq_diagnosis_codes])
    n_samples = len(seq_diagnosis_codes)
    n_diagnosis_codes = options['n_diagnosis_codes']
    maxlen = np.max(lengths)

    f_1 = 1e-5
    batch_diagnosis_codes = f_1 * np.ones((maxlen, n_samples, n_diagnosis_codes), dtype=np.float32)
    batch_mask = np.zeros((maxlen, n_samples), dtype=np.float32)

    for idx, c in enumerate(seq_diagnosis_codes):
        for x, subseq in zip(batch_diagnosis_codes[:, idx, :], c[:]):
            l = len(subseq)
            f_2 = float((l - f_1 * (4130 - l)) / l)
            x[subseq] = f_2

    for i in range(n_samples):
        max_visit = lengths[i] - 1
        batch_mask[max_visit, i] = 1

    batch_labels = np.array(seq_labels, dtype=np.int64)

    return batch_diagnosis_codes, batch_labels, batch_mask


def calculate_cost(model, data, options):
    batch_size = options['batch_size']
    n_batches = int(np.ceil(float(len(data[0])) / float(batch_size)))
    cost_sum = 0.0
    for index in range(n_batches):
        batch_diagnosis_codes = data[0][batch_size * index: batch_size * (index + 1)]
        batch_labels = data[1][batch_size * index: batch_size * (index + 1)]
        t_diagnosis_codes, t_labels, t_mask = pad_matrix(batch_diagnosis_codes, batch_labels, options)

        model_input = copy.copy(t_diagnosis_codes)
        for i in range(len(model_input)):
            for j in range(len(model_input[i])):
                idx = 0
                for k in range(len(model_input[i][j])):
                    model_input[i][j][k] = idx
                    idx += 1


        model_input = torch.LongTensor(model_input).cuda()
        t_labels = torch.LongTensor(t_labels).cuda()
        t_diagnosis_codes = torch.tensor(t_diagnosis_codes).cuda()

        logit = model(model_input, t_diagnosis_codes)
        loss = F.cross_entropy(logit, t_labels)
        cost_sum += loss.cpu().data.numpy()
    return cost_sum / n_batches

def write_data(file_path,data_name):
    pickle.dump(data_name,
                open(file_path, 'wb'))

