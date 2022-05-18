import numpy as np
import pickle
import operator
import copy
import torch
from copy import deepcopy

def log_write(MODEL_TYPE,INDICE_K,TAU,SECONDS):
    log_f = open('./Logs/GB_%s_k=%d_t=%s_s=%d.bak' % (MODEL_TYPE, INDICE_K, str(TAU), SECONDS), 'w+')

    return log_f

def write_data(file_path,data_name):
    pickle.dump(data_name,
                open(file_path, 'wb'))

def load_data(file_path):
    data = pickle.load(open(file_path, 'rb'))

    return data


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

def SetInsert_2D(ori_data,set_att_visit,set_att_code):
    set_data = deepcopy(ori_data)
    for visit,code in zip(set_att_visit,set_att_code):
        if str(visit) != str(()):
            set_data[visit] = SetInsert(ori_data[visit],[code])

    return set_data

def SetInsert_vect(person,set_):
    visit,code = FeatureTo2D(set_)
    set_data = SetInsert_2D(person,list(visit),list(code))
    return set_data

def SetInsert_1D(ori_data,visit,code):
    set_data = deepcopy(ori_data)
    if str(visit) != str(()):
        set_data[visit] = SetInsert(ori_data[visit],[code])

    return set_data

def MatrixToVector(data):
    vector = []
    for i in range(len(data)):
        a = np.array(data[i]) + i*len(data)
        vector = vector +list(a)
    return vector


def FeatureTo2D(feature_index):
    visit = np.array(feature_index) // 4130
    code = np.array(feature_index) % 4130

    return visit, code

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

def combinationSet(input):
    Com = []
    for i in range(1,len(input)+1):
        a = list(combinations(input,i))
        Com = Com + a
    return Com

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
