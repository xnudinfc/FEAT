import numpy as np
from tools import *
from MyCNN import *
import random
import torch.optim as optim
from torch.autograd import Variable
import time
from copy import deepcopy
import argparse

parser = argparse.ArgumentParser(description='EHR with LSTM in PyTorch ')  # 创建parser对象
parser.add_argument('--File_index', default=0, type=int, help='file index')
args = parser.parse_args()  # 解析参数，此处args是一个命名空间列表

torch.manual_seed(666)
torch.cuda.manual_seed(666)
random.seed(666)
TAU = 0.5
TopK = 1
SECONDS = 3600
Budget = 15
File_index = args.File_index
MODEL_TYPE = 'nonsub_UCB'
Algo_TYPE = ' FSGS' + '_' + str(File_index)
log_f = open('./Logs/%s/%s_Budget=%s_k=%d_t=%s_s=%d_test.bak' % (MODEL_TYPE, Algo_TYPE,str(Budget), TopK, str(TAU), SECONDS), 'w+')

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
if MODEL_TYPE == 'nonsub_UCB':
    net = Net_0D(num_uniqFeature).cuda()
    net.load_state_dict(torch.load('Output/net_weight_nopre_embed/1e-06.30'))
    net.eval()
elif MODEL_TYPE == 'sub':
    net = Net_0D(num_uniqFeature).cuda()
    net.load_state_dict(torch.load('Output/net_weight_nopre_embed_sub/1e-06.450'))
    net.eval()

CEloss = nn.CrossEntropyLoss().cuda()


def object1(g_min, S, alpha=0.1):
    return g_min + alpha * S / num_uniqFeature


def object2(prob, cardinality):
    return prob - 1 / (cardinality + 1)


def object3(prob, cardinality, alpha=0.01):
    return alpha * prob + cardinality


def Getpred(input=[], label=label):
    batch_attack_data = np.array([list(GetweightG(input, num_uniqFeature))])
    weight = torch.unsqueeze(torch.tensor(batch_attack_data), dim=1).cuda()

    logit = net(weight).cuda()
    pred_label = torch.max(logit[0].cpu().detach(), 0)[1].numpy()
    logit = logit[0][int(label)].cpu().detach().numpy()

    return logit, pred_label


def Getpred_M(input=[], label=1):
    weight = torch.unsqueeze(torch.tensor(input), dim=1).cuda()

    logit = net(weight).cuda()
    pred_value = logit.cpu().detach().numpy()[:, int(1 - label)]

    return pred_value


success_num = 0
success_data = []
danger_data = []
sample_index = []
success_label = []
NoAttack_num = 0
F = []
g = []
F_V = []
Total_iteration = 0
Total_targCode = 0
Total_time = 0
i = 0
for ori_data, ori_label in zip(data, label):
    ori_data = list(np.where(ori_data > 0.5)[0])
    time_start = time.time()
    i = i + 1

    robust_flag = 1  # (1 is robust, 0 is Noneed attack, -1 is not robust)
    print('-------------the number %d/%d----------------' % (i, len(label)), file=log_f, flush=True)
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

    Set_candidate = []  # the candidate set is S after selecting code process

    Set_target = [()]  # the target set is best chosen attck set_u_att under S
    Set_delet = []

    Set_c = [()]
    iteration = 0
    F_value_index = 0
    Set_residual = DiffElem(allCode, Set_candidate)
    set_best = [()]
    g_set_best = [()]
    F_best = 1 - pred_value_ori  # the best value of target F function).  it is different from the F_u(S) = F(S+u) - F(S)
    g_best = pred_value_ori  # the best value of target g function
    time_Dur = 0
    while robust_flag == 1 and len(Set_candidate)<= Budget and time_Dur <= SECONDS:
        iteration += 1
        # print(("----the %dth---")%(iteration), file=log_f, flush=True)

        g_set = []  # this is the list of the value of g(set_u_att).
        code_cand = []  # this is the list of selected code of each set_
        set_u_att = []  # the attack set after selected feature  u under the set_
        F_set = []  # this is the list of the value of F_u(set_u_att). This is to prepare random select
        F_S_new = []  # this is the list of the value of F(set_u_att)

        # Finally we want to save S and its coresponding F_u(S) value F_S
        # save Set_target and its coresponding g_u(S) value g_S
        batch_size = 4
        batch_num = len(Set_residual) // batch_size
        # batch_num = 2
        CodeMaxPred = []
        CodeMaxPred_index = []
        TopFeatures_index = []
        PRED = []
        # Set_pred = np.zeros([batch_num * batch_size])
        for set_ in Set_c:
            set_data = SetInsert(ori_data, set_)
            attack_matrix = GetAllAttck(set_data, Set_residual, num_uniqFeature)
            Set_pred = np.zeros([batch_num * batch_size])
            for index in range(batch_num):
                batch_attack_data = attack_matrix[batch_size * index: batch_size * (index + 1)]
                batch_attack_pred = Getpred_M(input=batch_attack_data)
                Set_pred[batch_size * index: batch_size * (index + 1)] = batch_attack_pred

            topk_feature_index = np.argsort(Set_pred)[-TopK:]
            CodeMaxPred.append(Set_pred[topk_feature_index[0]])  #
            TopFeatures_index.append(topk_feature_index)

        topk_set = np.argmax(CodeMaxPred)
        SetMax = Set_c[topk_set]
        FeatMax = Set_residual[TopFeatures_index[topk_set][0]]

        ScoreMax = CodeMaxPred[topk_set]

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
    if (i - NoAttack_num) != 0 and success_num != 0 :
        print("--- success Ratio: " + str(success_num / (i - NoAttack_num)) + " ---", file=log_f, flush=True)
        print("--- Mean Iteration: " + str(Total_iteration / success_num) + " ---", file=log_f,
              flush=True)
        print("--- Mean TargetCode: " + str(Total_targCode / success_num) + " ---", file=log_f,
              flush=True)
        print("--- Mean Time: " + str(Total_time / success_num) + " ---", file=log_f,
              flush=True)

print(TITLE)
print(TITLE, file=log_f, flush=True)
