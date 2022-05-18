import time
import torch
import torch.nn as nn
import rnn_tools
import rnn_model
from torch.autograd import Variable
from copy import deepcopy
import os
import pickle
import numpy as np
from math import pow
from itertools import combinations
from tools import *
import itertools
import random
import argparse
import math
from collections import Counter

parser = argparse.ArgumentParser(description='EHR with LSTM in PyTorch ')    #创建parser对象
parser.add_argument('--File_index', default=0, type=int, help='file index')
args=parser.parse_args()#解析参数，此处args是一个命名空间列表

torch.manual_seed(666)
torch.cuda.manual_seed(666)
random.seed(666)
TAU = 0.5
TopK = 40
SECONDS= 3600
Budget = 5
ucb_num =3
ALPHA = 0

File_index = args.File_index
MODEL_TYPE = 'nonsub'
Algo_TYPE = ' RG'

TITLE = '=== ' + MODEL_TYPE +Algo_TYPE+ ' target prob = ' + str(TAU) + ' k = ' \
        + str(TopK) + ' time = ' + str(SECONDS) + ' ==='

NUM_ATTACK_SAMPLES = 200

log_f = open('./Logs/%s_UCB/%s_budget=%s_ucb_num=%d_k=%d_ALPHA = %s_t=%s_s=%d.bak' % (MODEL_TYPE,Algo_TYPE, str(Budget),ucb_num, TopK,str(ALPHA),str(TAU), SECONDS), 'w+')


class Attacker(object):
    def __init__(self, options, emb_weights):
        print("Loading pre-trained classifier...", file=log_f, flush=True)

        self.model = rnn_model.LSTM(options, emb_weights).cuda()
        self.model2 = rnn_model.LSTM(options, emb_weights).cuda()

        if MODEL_TYPE == 'sub':
            self.model.load_state_dict(torch.load('./Classifiers/Submodular_lstm.49'))  # abs
            self.model2.load_state_dict(torch.load('./Classifiers/Submodular_lstm.49'))
        elif MODEL_TYPE == 'nonsub':
            self.model.load_state_dict(torch.load('./Classifiers/Nonsubmodular_lstm.42'))  # positive and negative
            self.model2.load_state_dict(torch.load('./Classifiers/Nonsubmodular_lstm.42'))
        elif MODEL_TYPE == 'weaksub':
            self.model.load_state_dict(torch.load('./Classifiers/Weaksubmodular_lstm.43'))  # weak abs
            self.model2.load_state_dict(torch.load('./Classifiers/weaksubmodular_lstm.43'))

        self.model.train()
        self.model2.eval()

        self.criterion = nn.CrossEntropyLoss()

    def classify(self, person,y):

        model_input, weight_of_embed_codes = self.input_handle(person)

        logit = self.model2(model_input, weight_of_embed_codes)

        pred = torch.max(logit[0].cpu().detach(), 0)[1].numpy()
        prob = logit[0][int(y)].cpu().detach().numpy()

        return pred, prob

    def classifyM(self, person,y):

        model_input, weight_of_embed_codes = self.input_handleM(person)

        logit = self.model2(model_input, weight_of_embed_codes)

        prob = logit.cpu().detach().numpy()[:,int(y)]

        return prob

    def input_handle(self, person):
        t_diagnosis_codes = rnn_tools.pad_matrix(person)
        model_input = deepcopy(t_diagnosis_codes)
        for i in range(len(model_input)):
            for j in range(len(model_input[i])):
                idx = 0
                for k in range(len(model_input[i][j])):
                    model_input[i][j][k] = idx
                    idx += 1

        model_input = Variable(torch.LongTensor(model_input))
        return model_input.transpose(0, 1).cuda(), torch.tensor(t_diagnosis_codes).transpose(0, 1).cuda()

    def input_handleM(self, person):
        t_diagnosis_codes,t_labels, batch_mask = rnn_tools.pad_matrix_M(person,[])
        model_input = copy.copy(t_diagnosis_codes)
        for i in range(len(model_input)):
            for j in range(len(model_input[i])):
                idx = 0
                for k in range(len(model_input[i][j])):
                    model_input[i][j][k] = idx
                    idx += 1

        model_input = Variable(torch.LongTensor(model_input).cuda())
        t_diagnosis_codes = torch.tensor(t_diagnosis_codes).cuda()

        return model_input, t_diagnosis_codes


    def forward_lstm(self, weighted_embed_codes, model):
        x = model.relu(weighted_embed_codes)
        x = torch.mean(x, dim=2)
        h0 = Variable(torch.FloatTensor(torch.randn(1, x.size()[1], x.size()[2])))
        c0 = Variable(torch.FloatTensor(torch.randn(1, x.size()[1], x.size()[2])))
        output, h_n = model.lstm(x, (h0, c0))
        embedding, attn_weights = model.attention(output.transpose(0, 1))
        x = model.dropout(embedding)  # (n_samples, hidden_size)

        logit = model.fc(x)  # (n_samples, n_labels)

        logit = model.softmax(logit)
        return logit


    def getMatrix(self,set_data,visit_R,code_R):
        attack_matrix = []
        for visit, code in zip(visit_R,code_R):
            data = deepcopy(set_data)
            if str(visit) != str(()):
                data[visit] = SetInsert(set_data[visit], [code])
            attack_matrix.append(data)
        return attack_matrix

    def RSelectCode(self,topk_set_feature, BestCode_index ,BestPredCode,SetPred_max_R_index, Set_c_visit,Set_c_code, ori_label,person):
        Select_visit = random.choice(topk_set_feature)
        Select_code = BestCode_index[Select_visit]
        Selectset_visit = Set_c_visit[int(SetPred_max_R_index[Select_visit])]
        Selectset_code = Set_c_code[int(SetPred_max_R_index[Select_visit])]

        set_att_visit = UnionEle(Selectset_visit, [Select_visit])
        set_att_code = UnionEle(Selectset_code, [Select_code])

        att_data = SetInsert_2D(person,set_att_visit,set_att_code)
        prob = BestPredCode[Select_visit]
        # pred, prob = self.classify(att_data,1-ori_label)

        return prob,Select_visit,Select_code,Selectset_visit,Selectset_code,set_att_visit,set_att_code,att_data

    def UCBV(self,round, pred_set_list, N,alpha):

        mean = np.mean(pred_set_list, axis=0)
        variation = np.var(pred_set_list, axis=0)
        delta = np.sqrt((alpha * variation * math.log(round)) / (N)) + (alpha * math.log(round) / (N))
        ucb = mean + delta

        return ucb

    def attack(self, person, y,Set_residual):

        batch_size = 1
        batch_num = len(Set_residual) // batch_size

        set_ = []
        set_data = SetInsert_vect(person, set_)
        visit_R, code_R = FeatureTo2D(Set_residual)
        attack_matrix = self.getMatrix(set_data, visit_R, code_R)

        Set_pred_R = np.zeros([batch_num * batch_size])
        for index in range(batch_num):
            batch_attack_data = attack_matrix[batch_size * index: batch_size * (index + 1)]
            batch_attack_prob = self.classifyM(batch_attack_data, 1 - y)
            Set_pred_R[batch_size * index: batch_size * (index + 1)] = batch_attack_prob


        return Set_pred_R

    def GetGradient(self,input, label):
        model_input, weight_of_embed_codes = self.input_handle(input)
        weight_of_embed_codes.requires_grad_()
        logit = self.model(model_input, weight_of_embed_codes)
        loss = self.criterion(logit, Variable(torch.LongTensor([label])).cuda())
        loss.backward()
        prob = logit.cpu().detach().numpy()[:, int(label)]

        grad = weight_of_embed_codes.grad.cpu().detach().numpy()[:,0,:]


        return grad, loss.cpu().detach().numpy(),prob

def UCBV(round, pred_set_list, N,alpha):

    mean = pred_set_list / N
    variation = (pred_set_list - mean) ** 2 / N
    delta = np.sqrt((alpha * variation * math.log(round)) / (N)) + (alpha * math.log(round) / (N))
    ucb = mean + delta

    return ucb


def main(emb_weights, training_file, validation_file,
         testing_file, n_diagnosis_codes, n_labels,
         batch_size, dropout_rate,
         L2_reg, n_epoch, log_eps, n_claims, visit_size, hidden_size,
         use_gpu, model_name):
    options = locals().copy()
    print("Loading dataset...", file=log_f, flush=True)
    test = rnn_tools.load_data(training_file, validation_file, testing_file)

    n_people = NUM_ATTACK_SAMPLES

    attacker = Attacker(options, emb_weights)

    n_success = 0
    n_fail = 0

    total_node_change = 0

    n_iteration = 0

    saving_time = {}

    attack_code_dict = {}

    NoAttack_num = 0
    success_num = 0
    success_data = []
    danger_data = []
    sample_index =[]
    success_label = []

    Total_iteration = 0
    Total_time = 0
    Total_change = 0

    F = []
    g = []
    F_V = []
    Total_iteration = 0
    Total_targCode = 0
    Total_Query = 0

    for i in range(0,200):
        print("-------- %d ---------" % (i), file=log_f, flush=True)

        person = test[0][i]

        y = test[1][i]

        n_visit = len(person)

        print('* Processing:%d/%d person, number of visit for this person: %d' % (i, n_people, n_visit), file=log_f,
              flush=True)

        # print("* Original: " + str(person), file=log_f, flush=True)

        print("  Original label: %d" % (y), file=log_f, flush=True)

        time_start = time.time()
        # changed_person, score, num_changed, success_flag, iteration, changed_pos = attacker.attack(person, y)
        robust_flag = 1
        orig_pred, orig_prob = attacker.classify(person, y)

        if orig_pred != y:
            NoAttack_num += 1
            print('ori_classifier predicts wrong!', file=log_f, flush=True)
            robust_tag = 0
            continue

        Set_candidate = []
        Set_delet = []

        Set_target = [()]
        Set_c = [()]

        F_best = 1 - orig_prob

        F_S =[]
        g_target = []
        F_value = []

        iteration = 0
        allCode = list(range(len(person)*4130))
        # Set_residual = DiffElem(allCode, Set_candidate)
        RN = 5  # this is the time to repeat all the random process. to get a more stable result.
        success = []
        num_armchain = []
        n_change = []
        time_success = []
        arm_chains = []
        arm_preds = [orig_pred]
        success_sample = []
        danger_sample = []
        set_data = person
        for n in range(RN):
            # print("random index :",n)
            start_random = time.time()

            arm_chain = []  # the candidate set is S after selecting code process
            arm_pred = []
            iteration = 0
            F_value_index = 0
            batch_size = 4

            pred_set_list = []
            time_Dur = 0
            robust_flag = 1

            while robust_flag == 1 and  len(Counter(arm_chain).keys())<= Budget and time_Dur <= SECONDS:

                SetGrad_max_R = (-100) * np.ones([len(person) * 4130])
                SetGrad_max_R_index = np.zeros([len(person) * 4130], dtype='int')
                grad_set, loss, prob = attacker.GetGradient(input=set_data, label=1 - y)
                grad = grad_set.reshape([len(person) * 4130])

                grad = abs(grad)
                SetGrad_max_R_index = np.argsort(grad)
                K_set = SetGrad_max_R_index[-TopK:]

                batch_num = len(K_set) // batch_size
                N = np.ones(len(K_set))

                pred_set_list = attacker.attack(set_data, y, K_set)

                ucb_loop = 0
                INDEX = []
                while robust_flag == 1 and ucb_loop <= ucb_num and len(Counter(arm_chain).keys())< Budget:
                    ucb_loop = ucb_loop + 1
                    iteration += 1
                    # print('K_set', K_set, file=log_f, flush=True)

                    ucb = UCBV(iteration, pred_set_list, N,ALPHA)
                    topk_feature_index = np.argsort(ucb)[-1]
                    INDEX.append(topk_feature_index)

                    Feat_max = K_set[topk_feature_index]
                    arm_chain.append(Feat_max)
                    set_data = SetInsert_vect(person, arm_chain)
                    new_pred, pred_max = attacker.classify(set_data, 1 - y)
                    arm_pred.append(pred_max)

                    n_add = np.eye(len(N))[topk_feature_index]
                    N += n_add

                    pred_set_list_add= np.zeros(len(K_set))
                    pred_set_list_add[topk_feature_index] = pred_max
                    pred_set_list = pred_set_list + pred_set_list_add

                    time_end = time.time()
                    time_Dur = time_end - start_random




                    if arm_pred[-1] > TAU:
                        success.append(1)
                        num_armchain.append(len(arm_chain))
                        n_change.append(len(Counter(arm_chain).keys()))
                        time_success.append(time_Dur)
                        arm_chains.append(arm_chain)
                        arm_preds.append(arm_pred)
                        robust_flag = 0
                        # print('arm_pred', arm_pred, file=log_f, flush=True)
                        # print('arm_chain', arm_chain, file=log_f, flush=True)
                        # print('attack success', file=log_f, flush=True)
                        break
                    if time_Dur > SECONDS:
                        print('The time is over', time_Dur, file=log_f, flush=True)
                        break
                if time_Dur > SECONDS:
                    print('The time is over', time_Dur, file=log_f, flush=True)
                    break
        arm_chains_sample = []
        arm_preds_sample = 0
        if np.sum(success) >= 1:
            # print('all random success attack in this sample')
            SR_sample = np.sum(success) / RN
            success_num += SR_sample
            AI_sample = np.average(num_armchain)
            AChange = np.average(n_change)
            AT_sample = np.average(time_success)

            arm_chains_sample = arm_chains
            arm_preds_sample = arm_preds

            Total_iteration += AI_sample
            Total_time += AT_sample
            Total_change += AChange
            success_data.append(success_sample)
            danger_data.append(danger_sample)

            Total_Query = Total_Query + (TopK + ucb_num) * AI_sample // (ucb_num)
            print("  Number of iterations for this: ", AI_sample, file=log_f, flush=True)
            print(" Time: ", AT_sample, file=log_f, flush=True)


        print("* SUCCESS Number NOW: %d " % (success_num), file=log_f, flush=True)
        print("* NoAttack Number NOW: %d " % (NoAttack_num), file=log_f, flush=True)

        print("--- Total Success Number: " + str(success_num) + " ---", file=log_f, flush=True)
        print("--- Total No Attack Number: " + str(NoAttack_num) + " ---", file=log_f, flush=True)
        if (i+1 - NoAttack_num) != 0 and success_num != 0:
            print("--- success Ratio: " + str(success_num / (i+1 - NoAttack_num)) + " ---", file=log_f, flush=True)
            print("--- Mean Iteration: " + str(Total_iteration / success_num) + " ---", file=log_f,
                  flush=True)
            print("--- Mean change: " + str(Total_change / success_num) + " ---", file=log_f,
                  flush=True)
            print("--- Mean Query: " + str(Total_Query / success_num) + " ---", file=log_f,
                  flush=True)
            print("--- Mean Time: " + str(Total_time / success_num) + " ---", file=log_f,
                  flush=True)


    print(TITLE)
    print(TITLE, file=log_f, flush=True)


if __name__ == '__main__':
    print(TITLE, file=log_f, flush=True)
    print(TITLE)
    # parameters
    batch_size = 5
    dropout_rate = 0.5
    L2_reg = 0.001  # 0.001
    log_eps = 1e-8
    n_epoch = 50
    n_labels = 2  # binary classification
    visit_size = 70
    hidden_size = 70
    n_diagnosis_codes = 4130
    n_claims = 504

    use_gpu = False
    model_name = 'lstm'

    trianing_file = './SourceData/hf_dataset_training.pickle'
    validation_file = './SourceData/hf_dataset_validation.pickle'
    testing_file = './SourceData/hf_dataset_testing_200.pickle'

    emb_weights_char = torch.load("./SourceData/PretrainedEmbedding.4")['char_embeddings.weight']
    emb_weights_word = torch.load("./SourceData/PretrainedEmbedding.4")['word_embeddings.weight']

    ##################

    map_char_idx = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '.': 10, 'E': 11,
                    'V': 12, 'VAC': 13}

    tree = pickle.load(open('./SourceData/hf_dataset_270_code_dict.pickle', 'rb'))
    map_codeidx_charidx = {}

    for k in tree.keys():
        codeidx = tree[k]
        charidx = []

        code = str(k)
        len_code = len(code)

        if len_code == 7:
            for c in code:
                charidx.append(map_char_idx[c])

        elif len_code == 6:

            if code[0] == 'V':
                charidx.append(map_char_idx[code[0]])
                charidx.append(map_char_idx['VAC'])
                for i in range(1, 6):
                    charidx.append(map_char_idx[code[i]])

            elif code[0] == 'E':
                charidx.append(map_char_idx[code[0]])
                for i in range(1, 6):
                    charidx.append(map_char_idx[code[i]])
                charidx.append(map_char_idx['VAC'])

            else:
                charidx.append(map_char_idx['VAC'])
                for i in range(6):
                    charidx.append(map_char_idx[code[i]])

        elif len_code == 5:

            if code[0] == 'V':
                charidx.append(map_char_idx[code[0]])
                charidx.append(map_char_idx['VAC'])
                for i in range(1, 5):
                    charidx.append(map_char_idx[code[i]])
                charidx.append(map_char_idx['VAC'])

            else:
                charidx.append(map_char_idx['VAC'])
                for i in range(5):
                    charidx.append(map_char_idx[code[i]])
                charidx.append(map_char_idx['VAC'])

        elif len_code == 4:
            for i in range(4):
                charidx.append(map_char_idx[code[i]])
            charidx.append(map_char_idx['VAC'])
            charidx.append(map_char_idx['VAC'])
            charidx.append(map_char_idx['VAC'])

        elif len_code == 3:
            if code[0] == 'V':
                charidx.append(map_char_idx[code[0]])
                charidx.append(map_char_idx['VAC'])
                charidx.append(map_char_idx[code[1]])
                charidx.append(map_char_idx[code[2]])
                charidx.append(map_char_idx['VAC'])
                charidx.append(map_char_idx['VAC'])
                charidx.append(map_char_idx['VAC'])
            else:
                charidx.append(map_char_idx['VAC'])
                charidx.append(map_char_idx[code[0]])
                charidx.append(map_char_idx[code[1]])
                charidx.append(map_char_idx[code[2]])
                charidx.append(map_char_idx['VAC'])
                charidx.append(map_char_idx['VAC'])
                charidx.append(map_char_idx['VAC'])

        map_codeidx_charidx[codeidx] = charidx

    codes_embedding = []

    for i in range(4130):
        chars = map_codeidx_charidx[i]

        char_code_embedding = []
        for c in chars:
            c_embedding = emb_weights_char[c].tolist()
            char_code_embedding.append(c_embedding)

        char_code_embedding = np.reshape(char_code_embedding, (-1))

        word_embedding = np.array(emb_weights_word[i])

        code_embedding = 0.5 * char_code_embedding + 0.5 * word_embedding

        codes_embedding.append(code_embedding)
    ##################

    emb_weights = torch.tensor(codes_embedding, dtype=torch.float)
    main(emb_weights, trianing_file, validation_file,
         testing_file, n_diagnosis_codes, n_labels,
         batch_size, dropout_rate,
         L2_reg, n_epoch, log_eps, n_claims, visit_size, hidden_size,
         use_gpu, model_name)
