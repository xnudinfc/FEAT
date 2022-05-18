import numpy as np
import pickle
import operator
import copy
import torch

def log_write(MODEL_TYPE,INDICE_K,TAU,SECONDS):
    log_f = open('./Logs/GB_%s_k=%d_t=%s_s=%d.bak' % (MODEL_TYPE, INDICE_K, str(TAU), SECONDS), 'w+')

    return log_f

def write_data(file_path,data_name):
    pickle.dump(data_name,
                open(file_path, 'wb'))

def load_data(file_path):
    data = pickle.load(open(file_path, 'rb'))

    return data


def create_unique_word_dict(text:list) -> dict:
    """
    A method that creates a dictionary where the keys are unique words
    and key values are indices
    """
    # Getting all the unique words from our text and sorting them alphabetically
    words = list(set(text))
    words.sort()

    # Creating the dictionary for the unique words
    unique_word_dict = {}
    for i, word in enumerate(words):
        unique_word_dict.update({
            word: i
        })

    return unique_word_dict


# Defining the window for context
def WordPair(window,all_strings):
    print('Creating a placeholder for the scanning of the word list')
    main_word = []
    context_word = []
    # Creating a context dictionary
    for i, word in enumerate(all_strings):
        for w in range(window):
            # Getting the context that is ahead by *window* words
            if i + 1 + w < len(all_strings):
                main_word.append(word)
                context_word.append(all_strings[(i + 1 + w)])
            # Getting the context that is behind by *window* words
            if i - w - 1 >= 0:
                main_word.append(word)
                context_word.append(all_strings[(i - w - 1)])

    return main_word,context_word

def stringTocode(string):
    code = np.array(operator.itemgetter(*string)(unique_word_dict))

def CodeNum():
    discreteData_path = 'dataset/HF_discrete_CodeSamples.pickle'
    discreteData = load_data(discreteData_path)
    length = []
    for i in discreteData:
        a = len(i)
        length.append(a)

    MAXL = max(length)#1121
    MINL = min(length)# 14

    return MAXL,MINL


def attack_data(discreteData_path,discreteData, label):
    discreteData_weight = np.where(discreteData == 0, 0.01, 0.99)

    data_input_norm = discreteData_weight
    # divide input data to train and test data
    pos_idx = np.where(label == 1)[0]
    neg_idx = np.where(label == 0)[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    attack_idx = pos_idx[int(float(len(pos_idx)) * 0.9):].tolist() + neg_idx[int(float(len(neg_idx)) * 0.9):].tolist()

    attack_data = np.array(data_input_norm)[np.array(attack_idx)]
    attack_label = label[attack_idx]
    pickle.dump([attack_data, attack_label], open(discreteData_path, 'wb'))

    return attack_data, attack_label

def AddOne(ori_data,diff):
    addone_data = []
    for code_add in diff:
        code_up = ori_data + [code_add]
        addone_data.append(code_up)

    return addone_data

def ResizeData(ori_data,num_uniqFeature):
    L_change = num_uniqFeature
    left = (L_change - len(ori_data)) //2
    right =  (L_change- len(ori_data)) - ((L_change - len(ori_data)) //2)
    resize_data = np.pad(ori_data, (left, right), 'constant', constant_values=num_uniqFeature)

    return resize_data

def Getweight(sample,num_uniqFeature):
    weight = np.zeros([num_uniqFeature])
    weight[sample] = 0.99
    weight = np.where(weight == 0, 0.01, weight)

    return weight

def GetweightG(sample,num_uniqFeature):
    weight = np.zeros([num_uniqFeature])
    weight[sample] = 0.99
    weight = np.where(weight == 0, 0.01, weight)
    return weight

def GetAllAttck(set_data,Set_residual,num_uniqFeature):
    weight = np.zeros([num_uniqFeature])
    weight[set_data] = 1
    matrix = np.eye(num_uniqFeature)[Set_residual]
    set_att = weight + matrix
    attack_matrix = np.where(set_att == 2, 0, set_att)
    attack_matrix = np.where(attack_matrix== 0, 0.01, 0.99)

    return attack_matrix

def MultiHot(data_,num_uniqFeature):
    MulHotData = np.zeros([num_uniqFeature])
    MulHotData[data_] = 1
    return MulHotData

def BFAttackmatrix(weight):
    w = copy.deepcopy(weight)
    w = np.where(w== 0.01, 0.98, w)
    w = np.where(w == 0.99, -0.98, w)
    # create the diagnal matrix using w vector
    matrix = weight + np.diag(w)
    return matrix

def GBAttackmatrix(weight,set):
    w = copy.deepcopy(weight)

    w = np.where(w== 0.01, 0.98, w)
    w = np.where(w == 0.99, -0.98, w)
    # create the diagnal matrix using w vector
    matrix = weight + np.diag(w)
    return matrix

def SameElem(a,b):
    same = list(set(a).intersection(set(b)))
    return same

def DiffElem(a,b):
    diff = list(set(a).difference(set(b)))
    return diff

def UnionEle(a,b):
    union = list(set(a).union(set(b)))
    return union

def DelSortList(a,b):
    l2 = list(set(a).difference(set(b)))
    l2.sort(key=a.index)
    return l2

def DeleteOne(ori_data):
    deleteone_data = []
    for code_delet in ori_data:
        code_change = copy.deepcopy(ori_data)
        code_change.remove(code_delet)
        deleteone_data.append(code_change)

    return deleteone_data

def SetInsert(ori_data,set_):
    union_ = UnionEle(ori_data, set_)
    same_ = SameElem(ori_data, set_)
    # get the attack data from the set_: # delete the same code and add the different code (ori_data and selected set_new)
    set_data_ = DiffElem(union_, same_)

    return set_data_

def NetInput(attack_code_add):
    attack_code_add = ResizeData(attack_code_add)
    attack_code_add = torch.LongTensor(attack_code_add).cuda()
    attack_code_add = torch.unsqueeze(attack_code_add, dim=0)
    attack_code_add = torch.unsqueeze(attack_code_add, dim=1)


    return attack_code_add



from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    a = chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    Pset = DiffElem(a,[()])
    return Pset
# def GetTopkCode():
#
#     return TopKcode

def DatesetInfo(discreteData_path):
    # discreteData_path = 'dataset/HF_discrete_CodeSamples.pickle'
    discreteData = np.array(load_data(discreteData_path))
    a_max = []
    b_min = []
    c_len = []
    for i in discreteData:
        a_max = [np.max(i)] + a_max
        b_min = [np.min(i)] + b_min
        c_len = [len(i)] + c_len

    Max = np.max(a_max)
    Min = np.min(b_min)
    Length_min = np.min(c_len)
    Length_max = np.max(c_len)

    return Max,Min,Length_min,Length_max

def GetwholeLine(data,w):
    data = [0,1,2]
    data = np.eye(11061)[data]
    np.matmul(data,w)

def minmax(a,k):
    index_max_0 = np.argsort(a, axis=0)[-1]
    value_max_0 = a[np.argsort(a, axis=0)[-1], np.array(range(a.shape[1]))]
    zero_num = len(np.where(value_max_0 == 0)[0])
    index_min_1 = np.argsort(value_max_0)
    if k > len(index_min_1) - zero_num:
        k = len(index_min_1) - zero_num
    select_feature_index_0 = index_max_0[index_min_1[-k:]]
    select_feature_index_1 = index_min_1[-k:]
    pred = a[select_feature_index_0,select_feature_index_1 ]

    return select_feature_index_0,select_feature_index_1,pred

def minmax_OMPGS(a,k):
    index_max_0 = np.argsort(a, axis=0)[-1]
    value_max_0 = a[np.argsort(a, axis=0)[-1], np.array(range(a.shape[1]))]
    zero_num = len(np.where(value_max_0 == 0)[0])
    index_min_1 = np.argsort(value_max_0)
    if k > len(index_min_1) - zero_num:
        k = len(index_min_1) - zero_num
    select_feature_index_0 = index_max_0[index_min_1[-k:]]
    select_feature_index_1 = index_min_1[-k:]
    pred = a[select_feature_index_0,select_feature_index_1 ]

    return select_feature_index_0,select_feature_index_1,pred
def minmax_FSGS(a,k):
    index_max_0 = np.argsort(a, axis=0)[-1]
    value_max_0 = a[np.argsort(a, axis=0)[-1], np.array(range(a.shape[1]))]
    zero_num = len(np.where(value_max_0 == 0)[0])
    index_min_1 = np.argsort(value_max_0)
    if k > len(index_min_1) - zero_num:
        k = len(index_min_1) - zero_num
    select_feature_index_0 = index_max_0[index_min_1[zero_num:k+zero_num]]
    select_feature_index_1 = index_min_1[zero_num:k+zero_num]
    pred = a[select_feature_index_0,select_feature_index_1 ]

    return select_feature_index_0,select_feature_index_1,pred

from collections import Counter
def Highfrequency_element(Word_sensitive):
    ini_list = Word_sensitive

    # printing initial ini_list
    print("initial list", str(ini_list))

    # sorting on bais of frequency of elements
    result = [item for items, c in Counter(ini_list).most_common()
              for item in [items] * c]

    # printing final result
    print("final list", str(result))