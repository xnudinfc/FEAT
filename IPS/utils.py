import numpy as np
import os
import pickle
import torch
from sklearn.model_selection import train_test_split


def make_dir(path):
    if os.path.isdir(path):
        pass
    else:
        os.mkdir(path)

# After attack, summarize the success rate, changed num and so on
def write_file(Dataset, Attack_Type, budget, algorithm, time_limit):
    log_f = open('./Logs/%s/MF_%s_%d_%a.bak' % (Dataset, Attack_Type, budget, algorithm), 'w+')
    TITLE = '=== ' + Dataset + Attack_Type + str(budget) + algorithm + ' time = ' + str(time_limit) + ' ==='
    print(TITLE, file=log_f, flush=True)
    directory = './Logs/%s/%s' % (Attack_Type, algorithm)
    print()
    print(directory)
    print(directory, file=log_f, flush=True)
    Algorithm = directory
    mf_process_temp = pickle.load(open(Algorithm + 'mf_process_%d.pickle' % budget, 'rb'))
    changed_set_process_temp = pickle.load(open(directory + 'changed_set_process_%d.pickle' % budget, 'rb'))
    robust_flag = pickle.load(open(directory + 'robust_flag_%d.pickle' % budget, 'rb'))
    query_num = pickle.load(open(directory + 'querynum_%d.pickle' % budget, 'rb'))
    time = pickle.load(open(directory + 'time_%d.pickle' % budget, 'rb'))
    iteration_file = pickle.load(open(directory + 'iteration_%d.pickle' % budget, 'rb'))
    funccall_all = pickle.load(open(directory + 'modified_funccall_%d.pickle' % budget, 'rb'))
    mf_process = []
    changed_set_process = []
    time_attack = []
    query_num_attack = []
    flip_changed_num = []
    iteration = []
    for j in range(len(robust_flag)):
        if robust_flag[j] == 0:
            mf_process.append(mf_process_temp[j])
            changed_set_process.append(changed_set_process_temp[j])
            time_attack.append(time[j])
            query_num_attack.append(query_num[j])
            flip_changed_num.append(len(changed_set_process_temp[j][-1]))
            iteration.append(iteration_file[j])

    sorted_flip_changed_num = np.sort(flip_changed_num)
    if sorted_flip_changed_num == np.array([]):
        change_medium = 0
    else:
        change_medium = sorted_flip_changed_num[len(flip_changed_num) // 2]

    print('success rate:', len(iteration) / len(mf_process_temp))
    print('average iteration:', np.mean(iteration))
    print('average changed code', np.mean(flip_changed_num))
    print('average time:', np.mean(time_attack))
    print('average query number', np.mean(query_num_attack))
    print('medium changed number', change_medium)
    print('clean test data accuracy:', len(robust_flag)/len(funccall_all))

    print('success rate:', len(iteration) / len(mf_process_temp), file=log_f, flush=True)
    print('average iteration:', np.mean(iteration), file=log_f, flush=True)
    print('average changed code', np.mean(flip_changed_num), file=log_f, flush=True)
    print('average time:', np.mean(time_attack), file=log_f, flush=True)
    print('average query number', np.mean(query_num_attack), file=log_f, flush=True)
    print('medium changed number', change_medium, file=log_f, flush=True)
    print('clean test data accuracy:', len(robust_flag) / len(funccall_all), file=log_f, flush=True)
    print('end')

# make some vector one hot vector
def one_hot_labels(y, n_labels):
    return torch.zeros(y.size(0), n_labels).long().scatter(1, y.unsqueeze(1).cpu(), 1).cuda()


def one_hot_samples(x, dataset):
    return torch.zeros(x.size(0), x.size(1), num_category[dataset]).long().scatter(2, x.unsqueeze(2).long().cpu(), 1)


def input_process(batch_diagnosis_codes, Dataset):
    t_diagnosis_codes = one_hot_samples(batch_diagnosis_codes, Dataset).cuda().float()
    return t_diagnosis_codes

def load_data():
    X_test = pickle.load(open('./dataset/IPSX_test.pickle', 'rb'))
    Y_test = pickle.load(open('./dataset/IPSY_test.pickle', 'rb'))

    return X_test, Y_test

num_category = {'IPS': 1104}
num_feature = {'IPS': 20}
num_samples = {'IPS': 4101}
num_avail_category = {'IPS': 1104}
num_classes = {'IPS': 3}