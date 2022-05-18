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
args=parser.parse_args() #解析参数，此处args是一个命名空间列表

torch.manual_seed(666)
torch.cuda.manual_seed(666)
random.seed(666)
TAU = 0.5
TopK = 10
SECONDS= 3600
Budget = 12
File_index = args.File_index
MODEL_TYPE = 'nonsub_UCB'
Algo_TYPE = ' OMPGS'
log_f = open('./Logs/%s/%s_Budget=%s_k=%d_t=%s_s=%d_test.bak'% (MODEL_TYPE,Algo_TYPE,str(Budget), TopK, str(TAU), SECONDS), 'w+')

TITLE = '=== ' + MODEL_TYPE + Algo_TYPE + ' target prob = ' + str(TAU) + ' k = ' \
        + str(TopK) + ' time = ' + str(SECONDS) + ' ==='

print(TITLE)
print(TITLE, file=log_f, flush=True)

attack_discreteData_path = 'dataset/5000_attackdata_0.2_200.pickle'
attack_discreteData = load_data(attack_discreteData_path)
data = attack_discreteData[0]
label = attack_discreteData[1]
num_uniqFeature = len(data[0])
# load model


# print('Load the CNN model', file=log_f, flush=True)
net = Net_0D(num_uniqFeature).cuda()
net.load_state_dict(torch.load('./Output/net_weight_nopre_embed/1e-06.30'))
net.eval()
net_grad = Net_0D(num_uniqFeature).cuda()
net_grad.load_state_dict(torch.load('./Output/net_weight_nopre_embed/1e-06.30'))
net_grad.eval()


CEloss = nn.CrossEntropyLoss().cuda()


def object1(g_min,S,alpha = 0.1):
    return g_min + alpha * S / num_uniqFeature

def object2(prob, cardinality):
    return prob - 1 / (cardinality + 1)

def object3(prob, cardinality,alpha = 0.01):
    return alpha*prob + cardinality

def SetInsert(ori_data,set_):
    union_ = UnionEle(ori_data, set_)
    same_ = SameElem(ori_data, set_)
    # get the attack data from the set_: # delete the same code and add the different code (ori_data and selected set_new)
    set_data_ = DiffElem(union_, same_)

    return set_data_

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

def Getpred(input=[], label=label):
    # net_pred= Net_0D(num_uniqFeature).cuda()
    # net_pred.load_state_dict(torch.load(path))
    batch_attack_data= np.array([list(GetweightG(input,num_uniqFeature))])
    weight = torch.unsqueeze(torch.tensor(batch_attack_data), dim=1).cuda()

    logit = net(weight).cuda()
    pred_label = torch.max(logit[0].cpu().detach(), 0)[1].numpy()
    logit = logit[0][int(label)].cpu().detach().numpy()

    return logit,pred_label

def RSelectCode(topk_set_feature,SetGrad_min_index,Set_c,ori_label):
    SelectCode = random.choice(topk_set_feature)
    Selectset = Set_c[int(SetGrad_min_index[SelectCode])]

    set_att = UnionEle(Selectset, [SelectCode])
    att_data = SetInsert(ori_data, set_att)
    logit,_ = Getpred(input=att_data, label=ori_label)

    return logit,SelectCode,Selectset,set_att

def RSelectCode_lossbase(topk_set_feature,setindex,Set_c,ori_label):
    SelectCode = random.choice(topk_set_feature)
    Selectset = Set_c[setindex]

    set_att = UnionEle(Selectset, [SelectCode])
    att_data = SetInsert(ori_data, set_att)
    logit,_ = Getpred(input=att_data, label=ori_label)

    return logit,SelectCode,Selectset

def FeatSelect(Set, topk_feature_index,label):
    feature = -1
    pred = -1
    for feature_index in topk_feature_index:

        set_att = UnionEle(Set, [feature_index])
        att_data = SetInsert(ori_data, set_att)
        logit_att,pred_att = Getpred(input=att_data, label=abs(1-label))

        if logit_att > pred:
            pred = logit_att
            feature = feature_index
        # print(feature_index,set_att,pred)
    return feature, pred

success_num = 0
success_data = []
danger_data = []
sample_index = []
success_label = []
NoAttack_num = 0
Total_time = 0
F = []
g = []
F_V = []
Total_iteration = 0
Total_targCode = 0
Total_query = 0
i = 0
for ori_data, ori_label in zip(data, label):
    ori_data = list(np.where(ori_data > 0.5)[0])

    time_start = time.time()
    i = i + 1

    robust_flag = 1 #(1 is robust, 0 is Noneed attack, -1 is not robust)
    print('========================the number %d/%d========================' % (i, len(label)), file=log_f, flush=True)
    pred_value_ori,pred_label_ori = Getpred(input=ori_data, label=ori_label)
    if pred_label_ori != ori_label:
        NoAttack_num += 1
        print('ori_classifier predicts wrong!', file=log_f, flush=True)
        robust_tag = 0
        continue

    allCode = list(range(num_uniqFeature))
    g_target = []  # The (:the best value of target F function).
    F_S = [] # The (S:the best value of target F function).  it is different from the F_u(S) = F(S+u) - F(S)
    F_value = []

    Set_candidate = []  # the candidate set is S after selecting code process

    Set_target = [()]  # the target set is best chosen attck set_u_att under S
    Set_delet = []

    Set_c = [()]
    iteration = 0
    query_num = 0
    F_value_index = 0

    F_best = 1-pred_value_ori  # the best value of target F function).  it is different from the F_u(S) = F(S+u) - F(S)
    g_best = pred_value_ori  # the best value of target g function

    while robust_flag == 1 and len(Set_target[-1])<= Budget and time.time()-time_start <= SECONDS:
        iteration +=1
        # print(("----the %dth---")%(iteration), file=log_f, flush=True)

        g_set = []  # this is the list of the value of g(set_u_att).
        code_cand = [] # this is the list of selected code of each set_
        set_u_att = []  # the attack set after selected feature  u under the set_
        F_set = []  # this is the list of the value of F_u(set_u_att). This is to prepare random select
        F_S_new = [] # this is the list of the value of F(set_u_att)

        # Finally we want to save S and its coresponding F_u(S) value F_S
        # save Set_target and its coresponding g_u(S) value g_S

        CodeMaxPred = []
        TopFeatures_index = []
        PRED = []
        for set_ in Set_c:
            # get the min_index of categorical and feature under set_ from Set_residual
            set_data_ = SetInsert(ori_data,set_)
            grad_set,loss,logit= GetGradient(input=set_data_, label=ori_label)

            grad_set = abs(grad_set)
            PRED.append(logit)
            topk_feature_index = DelSortList(DelSortList(list(np.argsort(grad_set)), Set_candidate), Set_delet)[-TopK:]  # for one subset, the max feature index in residual
            Top_feature_index, Top_pred = FeatSelect(set_, topk_feature_index,ori_label)

            CodeMaxPred.append(Top_pred)
            TopFeatures_index.append(Top_feature_index)

        topk_set = np.argmax(CodeMaxPred)
        SetMax = Set_c[topk_set]
        FeatMax = TopFeatures_index[topk_set]
        ScoreMax = CodeMaxPred[topk_set]

        query_num_each = TopK * math.pow(2, iteration-1)
        query_num = query_num + query_num_each

        if ScoreMax > F_best:
            F_best = ScoreMax
            set_att = UnionEle(SetMax, [FeatMax])
        else:
            F_best = F_best
            set_att = Set_target[-1]

        Set_candidate.append(FeatMax)

        Set_target.append(set_att)

        Set_residual = DiffElem(DiffElem(allCode, Set_candidate), Set_delet)
        Set_c = list(powerset(Set_candidate))

        if F_best > TAU:
            time_success = time.time() - time_start
            success_num += 1
            Total_iteration += len(Set_candidate)
            Total_targCode += len(set_att)
            Total_query += query_num
            Total_time += time_success
            print('attack success', file=log_f, flush=True)
            break

        time_end = time.time()
        time_Dur = time_end - time_start
        if time_Dur > SECONDS:
            print('The time is over', time_Dur, file=log_f, flush=True)
            break

    print("--- Total Success Number: " + str(success_num) + " ---", file=log_f, flush=True)
    print("--- Total No Attack Number: " + str(NoAttack_num) + " ---", file=log_f, flush=True)
    if (i - NoAttack_num) != 0 and success_num != 0:
        print("--- success Ratio: " + str(success_num / (i - NoAttack_num)) + " ---", file=log_f, flush=True)
        print("--- Mean Iteration: " + str(Total_iteration / success_num) + " ---", file=log_f,
              flush=True)
        print("--- Mean Query: " + str(Total_query/ success_num) + " ---", file=log_f,
              flush=True)
        print("--- Mean TargetCode: " + str(Total_targCode / success_num) + " ---", file=log_f,
              flush=True)
        print("--- Mean Time: " + str(Total_time / success_num) + " ---", file=log_f,
              flush=True)


print(TITLE)
print(TITLE, file=log_f, flush=True)
