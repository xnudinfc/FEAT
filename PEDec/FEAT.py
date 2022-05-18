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

parser = argparse.ArgumentParser(description='EHR with LSTM in PyTorch ')    #创建parser对象
parser.add_argument('--File_index', default=0, type=int, help='file index')
args=parser.parse_args()#解析参数，此处args是一个命名空间列表

torch.manual_seed(666)
torch.cuda.manual_seed(666)
random.seed(666)
TAU = 0.5

SECONDS= 3600
randK = 3000
Budget = 12
ucb_num = 5
ALPHA = 2
File_index = args.File_index
MODEL_TYPE = 'nonsub_UCB'
Algo_TYPE = ' RG_max' + 'ucb_num'+ str(ucb_num)
log_f = open('./Logs/%s/%s_Budget=%s_randK=%s_ALPHA=%s_t=%s_s=%d.bak'% (MODEL_TYPE,Algo_TYPE,str(Budget),str(randK),str(ALPHA),  str(TAU), SECONDS), 'w+')

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

def UCB(round,pred_set_list,N):

    mean = np.mean(pred_set_list,axis = 0)
    delta = np.sqrt((8 * math.log(round,10))/(N))
    ucb = mean + delta

    return ucb

def UCBV(round, pred_set_list, N,alpha):

    mean = pred_set_list / N
    variation = (pred_set_list - mean) ** 2 / N
    delta = np.sqrt((alpha * variation * math.log(round)) / (N)) + (alpha * math.log(round) / (N))
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


success_num = 0
success_data = []
danger_data = []
sample_index = []
success_label = []
arm_chains_samples = []
arm_preds_samples= []
NoAttack_num = 0
F = []
g = []
F_V = []
Total_iteration = 0
Total_time = 0
Total_change = 0
Total_Query = 0
i = 0
for ori_data, ori_label in zip(data, label):
    ori_data = list(np.where(ori_data > 0.5)[0])
    time_start = time.time()
    i = i + 1

    robust_flag = 1 #(1 is robust, 0 is Noneed attack, -1 is not robust)
    print('-------------the number %d/%d----------------' % (i, len(label)), file=log_f, flush=True)
    print('original label',ori_label, file=log_f, flush=True)
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
    num_armchain = []
    n_change = []
    time_success = []
    arm_chains = []
    arm_preds = [pred_value_ori]
    success_sample = []
    danger_sample = []
    batch_size = 4
    for n in range(RN):
        # print("random index :",n)
        start_random = time.time()
        set_data = ori_data
        WRS_loop = 0
        time_Dur = 0

        arm_chain = []  # the candidate set is S after selecting code process
        arm_pred = []
        iteration = 0
        F_value_index = 0
        batch_size = 2
        robust_flag = 1
        pred_set_list = []
        set_data = ori_data
        while robust_flag == 1 and len(Counter(arm_chain).keys())<= Budget and time_Dur <= SECONDS:
            # get weighted random sampling randK arms based on the bias
            grad_set, loss, logit = GetGradient(input=set_data, label=ori_label)

            Onecode_pred = np.absolute(grad_set)  # max: 0.0005, mean =  8.659937845686727e-06, min: 0
            SetGrad_max_R_index = np.argsort(grad_set)
            K_set = SetGrad_max_R_index[-randK:]

            batch_num = len(K_set) // batch_size
            N = np.ones(batch_num * batch_size)

            attack_matrix = GetAllAttck(set_data, K_set, num_uniqFeature)
            pred_set_list = np.zeros([batch_num * batch_size])

            for index in range(batch_num):
                batch_attack_data = attack_matrix[batch_size * index: batch_size * (index + 1)]
                batch_attack_pred = Getpred_M(input=batch_attack_data)
                pred_set_list[batch_size * index: batch_size * (index + 1)] = batch_attack_pred

            ucb_loop = 0
            INDEX = []
            while robust_flag == 1 and ucb_loop <= ucb_num and time_Dur <= SECONDS:
                ucb_loop = ucb_loop + 1
                iteration += 1

                ucb = UCBV(iteration, pred_set_list, N,ALPHA)
                topk_feature_index = np.argsort(ucb)[-1]
                INDEX.append(topk_feature_index)

                Feat_max = K_set[topk_feature_index]
                arm_chain.append(Feat_max)
                set_data = SetInsert(ori_data, arm_chain)
                pred_max, pred_label = Getpred(input=set_data, label=1-ori_label)
                arm_pred.append(pred_max)

                n_add = np.eye(len(N))[topk_feature_index]
                N += n_add

                pred_set_list_add = np.zeros(len(K_set))
                pred_set_list_add[topk_feature_index] = pred_max
                pred_set_list = pred_set_list + pred_set_list_add

                time_end = time.time()
                time_Dur = time_end - start_random

                # print("ucb_loop", ucb_loop, file=log_f, flush=True)
                # print("arm_chain", arm_chain, file=log_f, flush=True)
                # # print('K_set', K_set, file=log_f, flush=True)
                # # print('arm_chain',arm_chain , file=log_f, flush=True)
                # print('arm_pred', arm_pred, file=log_f, flush=True)
                if pred_max > TAU:
                    success.append(1)
                    num_armchain.append(len(arm_chain))
                    n_change.append(len(Counter(arm_chain).keys()))
                    time_success.append(time_Dur)
                    arm_chains.append(arm_chain)
                    arm_preds.append(arm_pred)

                    robust_flag = 0
                    # print("ucb_loop",ucb_loop, file=log_f, flush=True)
                    # print("arm_chain",arm_chain, file=log_f, flush=True)
                    # # print('K_set', K_set, file=log_f, flush=True)
                    # # print('arm_chain',arm_chain , file=log_f, flush=True)
                    # print('arm_pred',arm_pred , file=log_f, flush=True)
                    # # print('attack success', file=log_f, flush=True)
                    break

                if time_Dur > SECONDS:
                    print('The time is over', time_Dur, file=log_f, flush=True)
                    break
            if time_Dur > SECONDS:
                print('The time is over', time_Dur, file=log_f, flush=True)
                break

    if np.sum(success) >= 1:
        # print('all random success attack in this sample')
        SR_sample = np.sum(success)/RN
        success_num += SR_sample
        AI_sample = np.average(num_armchain)
        AChange = np.average(n_change)
        AT_sample = np.average(time_success)

        arm_chains_sample = arm_chains
        arm_preds_sample = arm_preds

        sample_index.append(i)
        success_label.append(ori_label)
        for d in arm_chains_sample:
            success_sample.append(SetInsert(ori_data, d))
            danger_sample.append(SetInsert(ori_data, d[:-1]))

        Total_iteration += AI_sample
        Total_time += AT_sample
        Total_change += AChange
        Total_Query = Total_Query + (randK + ucb_num) * AI_sample // (ucb_num)
        success_data.append(success_sample)
        danger_data.append(danger_sample)

        print("  Number of iterations for this: ", AI_sample, file=log_f, flush=True)
        print(" Time: ", AT_sample, file=log_f, flush=True)

    print("* SUCCESS Number NOW: %d " % (success_num), file=log_f, flush=True)
    print("* NoAttack Number NOW: %d " % (NoAttack_num), file=log_f, flush=True)

    print("--- Total Success Number: " + str(success_num) + " ---", file=log_f, flush=True)
    print("--- Total No Attack Number: " + str(NoAttack_num) + " ---", file=log_f, flush=True)
    if (i - NoAttack_num) != 0 and success_num !=0:
        print("--- success Ratio: " + str(success_num / (i - NoAttack_num)) + " ---", file=log_f, flush=True)
        print("--- Mean Iteration: " + str(Total_iteration / success_num) + " ---", file=log_f,
              flush=True)
        print("--- Mean Query: " + str(Total_Query / success_num) + " ---", file=log_f,
              flush=True)
        print("--- Mean Change: " + str(Total_change / success_num) + " ---", file=log_f,
              flush=True)
        print("--- Mean Time: " + str(Total_time / success_num) + " ---", file=log_f,
              flush=True)


print(TITLE,)
print(TITLE, file=log_f, flush=True)
