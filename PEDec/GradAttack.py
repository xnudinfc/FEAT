import numpy as np
from tools import *
from MyCNN import *
import random
import torch.optim as optim
from torch.autograd import Variable
import time
from copy import deepcopy
import argparse
import math
from copy import copy
parser = argparse.ArgumentParser(description='EHR with LSTM in PyTorch ')    #创建parser对象
parser.add_argument('--File_index', default=0, type=int, help='file index')
args=parser.parse_args()#解析参数，此处args是一个命名空间列表

torch.manual_seed(666)
torch.cuda.manual_seed(666)
random.seed(666)
TAU = 0.5

SECONDS= 3600
randK = 8
Budget = 10
File_index = args.File_index
MODEL_TYPE = 'nonsub_UCB'
Algo_TYPE = ' GA_' +'Topk=' + str(randK)
log_f = open('./Logs/%s/%s_budget=%s_t=%s_s=%d.bak'% (MODEL_TYPE,Algo_TYPE, str(Budget), str(TAU), SECONDS), 'w+')

TITLE = '=== ' + MODEL_TYPE + Algo_TYPE + ' target prob = ' + str(TAU) +' time = ' + str(SECONDS) + ' ==='

print(TITLE)
print(TITLE, file=log_f, flush=True)

attack_discreteData_path = 'dataset/5000_attackdata_0.2_200.pickle'
attack_discreteData = load_data(attack_discreteData_path)
data = attack_discreteData[0]
label = attack_discreteData[1]
num_uniqFeature = len(data[0])
# load model
# print('Load the CNN model', file=log_f, flush=True)
if MODEL_TYPE == 'nonsub_UCB':
    net = Net_0D(num_uniqFeature).cuda()
    net.load_state_dict(torch.load('Output/net_weight_nopre_embed/1e-06.30'))
    net.eval()
    net_grad = Net_0D(num_uniqFeature).cuda()
    net_grad.load_state_dict(torch.load('./Output/net_weight_nopre_embed/1e-06.30'))
    net_grad.eval()
elif MODEL_TYPE == 'sub':
    net = Net_0D(num_uniqFeature).cuda()
    net.load_state_dict(torch.load('Output/net_weight_nopre_embed_sub/1e-06.450'))
    net.eval()

CEloss = nn.CrossEntropyLoss().cuda()

def object1(g_min,S,alpha = 0.1):
    return g_min + alpha * S / num_uniqFeature

def object2(prob, cardinality):
    return prob - 1 / (cardinality + 1)

def object3(prob, cardinality,alpha = 0.01):
    return alpha*prob + cardinality


def Getpred(input=[], label=label):
    batch_attack_data= np.array([list(GetweightG(input,num_uniqFeature))])
    weight = torch.unsqueeze(torch.tensor(batch_attack_data), dim=1).cuda()

    logit = net(weight).cuda()
    pred_label = torch.max(logit[0].cpu().detach(), 0)[1].numpy()
    logit = logit[0][int(label)].cpu().detach().numpy()

    return logit,pred_label

def GetGradient(input=[], label=label):
    # net_grad=Net_0D(num_uniqFeature).cuda()
    # net_grad.load_state_dict(torch.load(path))
    batch_attack_data= np.array([list(GetweightG(input,num_uniqFeature))])
    weight = torch.unsqueeze(torch.tensor(batch_attack_data), dim=1).cuda()
    weight.requires_grad_()
    logit_ = net_grad(weight).cuda()
    loss = CEloss(logit_, Variable(torch.LongTensor([abs(1-label)])).cuda())
    loss.backward()
    grad = weight.grad.cpu().detach().numpy()[0,0,:]
    logit = logit_[0][int(abs(1-label))].cpu().detach().numpy()

    return grad,loss.cpu().detach().numpy(),logit

def Getpred_M(input=[],label=1):
    weight = torch.unsqueeze(torch.tensor(input), dim=1).cuda()

    logit = net(weight).cuda()
    pred_value = logit.cpu().detach().numpy()[:,int(1-label)]

    return pred_value

def UCB(round,Set_pred_R,N):

    mean = np.mean(pred_set_list,axis = 0)
    delta = np.sqrt((8 * math.log(round,10))/(N))
    ucb = mean + delta

    return ucb


from collections import Counter

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def removeElements(lst, k):
    counted = Counter(lst)

    temp_lst = []
    for el in counted:
        if counted[el] == k:
            temp_lst.append(el)

    res_lst = []
    for el in lst:
        if el not in temp_lst:
            res_lst.append(el)

    return (res_lst)


def attack_set( words, poses,ori_label,select_words):
    words = GetweightG(words,num_uniqFeature)
    candidates = [words]
    for pos in poses:
        current_candidates = copy(candidates)
        for cand in candidates:
            corrupted = copy(cand)
            if corrupted[pos] == 0.01:
                corrupted[pos] = 0.99
            else:
                corrupted[pos] = 0.01
            current_candidates.append(corrupted)
        candidates = copy(current_candidates)
    if len(candidates) != 1:
        batch_attack_pred = Getpred_M(input=candidates, label=ori_label)
        best_candidate_id = np.argmax(batch_attack_pred)
        new_data = candidates[best_candidate_id]
        pred_prob = batch_attack_pred[best_candidate_id]
    else:
        new_data = words
        pred_prob = Getpred_M(input=new_data, label=ori_label)
        print('empty candidates!')

    return new_data, pred_prob

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)
success_num = 0
success_data = []
time_success = []


danger_data = []
sample_index = []
success_label = []

num_armchain = []
num_search = []
arm_chains = []

arm_preds= []
NoAttack_num = 0
changed_words_all = []
F = []
g = []
F_V = []
Total_iteration = 0
Total_time = 0
Total_query = 0
count = 0
for ori_data, ori_label in zip(data, label):
    ori_data = list(np.where(ori_data > 0.5)[0])
    time_start = time.time()
    count = count + 1

    robust_flag = 1 #(1 is robust, 0 is Noneed attack, -1 is not robust)
    print('-------------the number %d/%d----------------' % (count, len(label)), file=log_f, flush=True)
    pred_value_ori, pred_label_ori = Getpred(input=ori_data, label=ori_label)
    if pred_label_ori != ori_label:
        NoAttack_num += 1
        print('ori_classifier predicts wrong!', file=log_f, flush=True)
        robust_tag = 0
        continue

    allCode = list(range(num_uniqFeature))
    g_target = []  # The (:the best value of target F function).
    F_S = []  # The (S:the best value of target F function).  it is different from the F_u(S) = F(S+u) - F(S)
    F_value = []

    RN = 5  # this is the time to repeat all the random process. to get a more stable result.
    success = []

    # arm_preds = [pred_value_ori]
    success_sample = []
    danger_sample = []
    batch_size = 4

    # performance index parameter
    # 1 success = [0,1,1,1,1]
    # 2 code number with success = [2,3,4,5,6]
    # 3 time computation with success = [2,3,4,5,6]
    # 4 search ability = [code number with success] /[time computation with success]

    iteration = 0
    F_value_index = 0
    pred_set_list = []
    new_data_ = ori_data
    best_score = 1 - pred_value_ori
    changed_words = []
    time_Dur = 0
    st = time.time()
    n_change = 0
    select_words = []
    before = GetweightG(ori_data, num_uniqFeature)
    Candidate = []
    while robust_flag == 1 and n_change <= Budget and time_Dur <= SECONDS:
        iteration += 1

        # get weighted random sampling randK arms based on the bias
        grad_set, loss, logit = GetGradient(input=new_data_, label=ori_label)

        Onecode_pred = np.absolute(grad_set)  # max: 0.0005, mean =  8.659937845686727e-06, min: 0
        if len(select_words) !=0:
            for i in select_words:
                Onecode_pred[i] = -10000
        Onecode_pred_nonzero_index = np.where(Onecode_pred != 0)[0]  # only 820 arms is not zero
        Onecode_sig_pred = Onecode_pred
        if len(Onecode_pred_nonzero_index) < randK:
            K_set = np.argsort(Onecode_sig_pred)[-len(Onecode_pred_nonzero_index):]
        else:
            K_set = np.argsort(Onecode_sig_pred)[-randK:]
        Candidate = Candidate + list(K_set)
        new_data, pred_prob = attack_set(new_data_, K_set, ori_label,select_words)

        n_change = sum([0 if before[i] == new_data[i] else 1 for i in range(len(new_data))])
        if n_change > Budget:
            n_change = sum([0 if before[i] == select_words[i] else 1 for i in range(len(select_words))])
            break
        if pred_prob > best_score:
            best_score = pred_prob
            new_data_ = np.where(new_data == 0.99)[0]

            select_words = []
            for ind in range(len(new_data)):
                if before[ind] != new_data[ind]:
                    select_words.append(ind)
        else:
            for k in list(K_set):
                select_words.append(k)

        # print("best score",best_score,select_words)
        time_Dur = time.time() - st
        if best_score > TAU:
            robust_flag = 0

        if best_score > TAU:
            for ind in range(len(new_data)):
                if before[ind] != new_data[ind]:
                    changed_words.append(ind)

            a = 0
            for j in range(1, randK + 1):
                C_value = j * nCr(randK, j)
                a = a + C_value
            query_num = a * iteration

            success_num = success_num + 1
            time_success.append(time_Dur)
            Total_iteration = Total_iteration + iteration
            Total_query += query_num
            changed_words_all.append(changed_words)
            arm_chains.append(changed_words)
            arm_preds.append(best_score)
            num_armchain.append(len(changed_words))
            num_search.append(len(Candidate))
            break

        if time_Dur > SECONDS:
            print('The time is over', time_Dur, file=log_f, flush=True)
            break



    print("  Number of iterations for this: ", iteration, file=log_f, flush=True)
    print(" Time: ",time_Dur, file=log_f, flush=True)

    print("* SUCCESS Number NOW: %d " % (success_num), file=log_f, flush=True)
    print("* NoAttack Number NOW: %d " % (NoAttack_num), file=log_f, flush=True)

    if (count - NoAttack_num) != 0 and success_num != 0:
        print("--- success Ratio: " + str(success_num / (count - NoAttack_num)) + " ---", file=log_f, flush=True)
        print("--- Mean Iteration: " + str(Total_iteration / success_num) + " ---", file=log_f,
              flush=True)
        print("--- Mean Query: " + str(Total_query/ success_num) + " ---", file=log_f,
              flush=True)
        print("--- Mean num_searched: " + str(np.sum(num_search) / success_num) + " ---", file=log_f,
              flush=True)
        print("--- Mean Time: " + str(np.sum(time_success) / success_num) + " ---", file=log_f,
              flush=True)
        print("--- Mean Code: " + str(np.sum(num_armchain) / success_num) + " ---", file=log_f,
              flush=True)


print(TITLE,)
print(TITLE, file=log_f, flush=True)
