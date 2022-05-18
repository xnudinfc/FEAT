from __future__ import print_function
import logging
import argparse
import math
from copy import copy
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
import random
import pickle
from collections import Counter
import torch
import os
import copy

NGRAM = 3
TAU = 0.5
N_REPLACE = 20  # len(doc)//N_REPLACE
SubK_ratio = 0.3
SECONDS = 1000
Budget = 6


# from gensim.test.utils import common_texts
# model = Word2Vec('I am happy today and very efficient', size=300, window=5, min_count=1, workers=4)
# model.wv.save_word2vec_format('token_vec_300.txt', binary=False)
# a = KeyedVectors.load_word2vec_format('', binary=False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(666)
torch.cuda.manual_seed(666)
random.seed(666)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='LSTM', type=str, help='model: either CNN or LSTM')
    parser.add_argument('--test_path', action='store', default='./mal_test_funccall.pickle', type=str,
                        dest='test_path',
                        help='Path to test.txt data')
    parser.add_argument('--max_size', default=20000, type=int,
                        help='max amount of transformations to be processed by each iteration')
    parser.add_argument('--ucbloop', default=20, type=int,
                        help='ucb_loop')
    parser.add_argument('--alpha', default=8, type=int,
                        help='alpha')

    return parser.parse_args()


class RNN(nn.Module):
    def __init__(self):
        # def __init__(self, emb_weights):
        super(RNN, self).__init__()
        dropout_rate = 0.5
        input_size = 70
        hidden_size = 70
        n_labels = 3
        n_diagnosis_codes = 1104
        self.num_layers = 1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=self.num_layers, batch_first=False)
        self.fc = nn.Linear(hidden_size, n_labels)
        self.attention = SelfAttention(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.embed = torch.nn.Embedding(n_diagnosis_codes, input_size)
        # self.embed.weight = nn.Parameter(torch.FloatTensor(emb_weights))

    # overload forward() method
    def forward(self, x, weight_of_x):
        x = x.long()
        x = self.embed(x)  # (n_visits, n_samples, visit_size)
        # multiply by weight here:
        weight_of_x = torch.unsqueeze(weight_of_x, dim=3)
        x = x * weight_of_x
        x = self.relu(x)

        x = torch.mean(x, dim=2)

        output, h_n = self.lstm(x)

        x, attn_weights = self.attention(output.transpose(0, 1))

        x = self.dropout(x)

        logit = self.fc(x)

        logit = self.softmax(logit)
        return logit


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        # (B, L, H) -> (B , L, 1)
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights


def pad_matrix(seq_diagnosis_codes, seq_labels, n_diagnosis_codes):
    lengths = np.array([len(seq) for seq in seq_diagnosis_codes])
    n_samples = len(seq_diagnosis_codes)
    maxlen = np.max(lengths)

    f_1 = 1e-5
    batch_diagnosis_codes = f_1 * np.ones((maxlen, n_samples, n_diagnosis_codes), dtype=np.float32)

    for idx, c in enumerate(seq_diagnosis_codes):
        for x, subseq in zip(batch_diagnosis_codes[:, idx, :], c[:]):
            l = 1
            f_2 = float((l - f_1 * (n_diagnosis_codes - l)) / l)
            x[subseq] = f_2

    batch_labels = np.array(seq_labels, dtype=np.int64)

    return batch_diagnosis_codes, batch_labels


class Attacker(object):
    ''' main part of the attack model '''

    def __init__(self, best_parameters_file, opt):
        self.opt = opt
        # self.DELTA_W = int(opt.word_delta) * 0.1
        self.model = RNN()
        # self.model = Net(dropout=False)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.load_state_dict(torch.load(best_parameters_file, map_location='cpu'))
        self.model.eval()
        logging.info("Initializing vocabularies...")
        print("Initializing vocabularies...")
        # to compute the gradient, we need to set up the optimizer first
        self.criterion = nn.CrossEntropyLoss().to(device)

    def word_paraphrase_nocombine(self, new_words, orig_prob_new, poses, category_set, y):
        index_candidate = new_words[poses]
        pred_candidate = np.array(len(poses)*[0.0])

        candidates = []
        for i, candi_sample_index in enumerate(poses):
            corrupted = copy.deepcopy(new_words)
            corrupted[candi_sample_index] = category_set[candi_sample_index]
            candidates.append(corrupted)

        batch_diagnosis_codes = candidates
        batch_labels = [y] * len(batch_diagnosis_codes)
        t_diagnosis_codes, t_labels = pad_matrix(batch_diagnosis_codes, batch_labels, 1104)

        model_input = copy.deepcopy(t_diagnosis_codes)
        for idx in range(len(model_input)):
            for j in range(len(model_input[idx])):
                temp = 0
                for k in range(len(model_input[idx][j])):
                    model_input[idx][j][k] = temp
                    temp += 1

        model_input = torch.FloatTensor(model_input).cuda()
        t_diagnosis_codes = torch.tensor(t_diagnosis_codes).cuda()
        # t_diagnosis_codes = torch.autograd.Variable(t_diagnosis_codes.data, requires_grad=True)

        pred_probs = self.model(model_input, t_diagnosis_codes)
        result = 1 - pred_probs[:, y]
        pred_prob = result.cpu().detach().numpy()

        pred_candidate = np.where(pred_prob >= pred_candidate, pred_prob, pred_candidate)
        index_candidate = np.where(pred_prob >= pred_candidate, category_set[poses], index_candidate)

        return index_candidate, pred_candidate

    def UCBV(self, round, pred_set_list, N):

        mean = pred_set_list / N
        variation = (pred_set_list - mean) ** 2 / N
        delta = np.sqrt((self.opt.alpha * variation * math.log(round)) / (N)) + (self.opt.alpha * math.log(round) / (N))
        ucb = mean + delta

        return ucb

    def New_Word(self, words, arm_chain, list_closest_neighbors):
        new_word = copy.deepcopy(words)
        if len(arm_chain) == 0:
            pass
        else:
            for pos in arm_chain:
                new_word[pos[0]] = pos[1]
                # if pos[1] != 0:
                #     # new_word[pos[0]] = list_closest_neighbors[pos[0]][pos[1] - 1]
                #     new_word[pos[0]] = list_closest_neighbors[pos[1] - 1]
        return new_word

    def input_handle(self, funccall, y):  # input:funccall, output:(seq_len,n_sample,m)[i][j][k]=k,
        funccall = [funccall]
        y = [y]
        t_diagnosis_codes, _ = pad_matrix(funccall, y, 1104)
        model_input = copy.deepcopy(t_diagnosis_codes)
        for i in range(len(model_input)):
            for j in range(len(model_input[i])):
                idx = 0
                for k in range(len(model_input[i][j])):
                    model_input[i][j][k] = idx
                    idx += 1

        model_input = torch.FloatTensor(model_input).cuda()
        return model_input, torch.tensor(t_diagnosis_codes).cuda()

    def classify(self, funccall, y):
        model_input, weight_of_embed_codes = self.input_handle(funccall, y)
        logit = self.model(model_input, weight_of_embed_codes)
        logit = logit.cpu()
        pred = torch.max(logit, 1)[1].view((1,)).data.numpy()
        logit = logit.data.cpu().numpy()
        g = logit[0][y]

        return g, pred

    def Grad_index(self, words, list_closest_neighbors, changed_pos, y):
        self.model.train()
        model_input, t_diagnosis_codes = self.input_handle(words, y)
        t_diagnosis_codes = torch.tensor(t_diagnosis_codes).cuda()
        t_diagnosis_codes = torch.autograd.Variable(t_diagnosis_codes.data, requires_grad=True)
        output = self.model(model_input, t_diagnosis_codes)

        if torch.cuda.is_available():
            loss = self.criterion(output, torch.autograd.Variable(torch.LongTensor([y])).to(device))
        else:
            loss = self.criterion(output, torch.autograd.Variable(torch.LongTensor([y])))
        loss.backward()
        grad = t_diagnosis_codes.grad.cpu().data
        grad = abs(grad)
        grad = torch.reshape(grad, (grad.size(0), -1))
        max_grad, max_category = torch.max(grad, dim=1)

        return max_grad.cpu().data, max_category.cpu().data

    def attack(self, count, words, y, NoAttack, log_f, N_REPLACE, SubK_ratio):
        # check if the value of this doc to be right
        orig_prob, orig_pred = self.classify(words, y)
        pred, pred_prob = orig_pred, orig_prob
        if not (pred == y):  # attack success
            print(" this original samples predict wrong")
            NoAttack = NoAttack + 1
            print(pred, y)
            return 0, 0, [], 0, 0, 0, 0, NoAttack
        best_score = 1 - pred_prob

        # now word level paraphrasing
        ## step1: find the neighbor for all the words.
        list_closest_neighbors = range(1104)

        changed_pos = set()
        iteration = 0
        recompute = True
        n_change = 0
        lword = 20
        changed_words = []

        allCode = range(lword)
        RN = 1  # this is the time to repeat all the random process. to get a more stable result.
        success = []
        num_armchain = []
        n_change = []
        time_success = []
        arm_chains = []
        arm_preds = [pred_prob]

        if len(allCode) <= N_REPLACE:
            N_REP = len(allCode)
        else:
            N_REP = N_REPLACE

        for n in range(RN):
            # print("random index :",n)
            start_random = time.time()
            iteration = 0
            WRS_loop = 0
            time_Dur = 0
            arm_chain = []  # the candidate set is S after selecting code process
            arm_pred = []
            robust_flag = 1
            changed_pos = []
            while robust_flag == 1 and len(Counter(arm_chain).keys()) <= Budget and time_Dur <= SECONDS:

                new_words = self.New_Word(words, arm_chain, list_closest_neighbors)
                orig_prob_new, orig_pred_new = self.classify(new_words, y)

                grad_set, category_set = self.Grad_index(new_words, list_closest_neighbors, changed_pos, y)
                self.model.eval()
                if torch.sum(grad_set) != 0:
                    WeightProb = np.array(grad_set / torch.sum(grad_set))
                else:
                    WeightProb = np.array([0.05] * 20)

                len_nonzero = len(np.nonzero(WeightProb)[0])
                if len_nonzero < N_REP:
                    K_set = np.random.choice(range(20), size=len_nonzero, replace=False, p=WeightProb)
                else:
                    K_set = np.random.choice(range(20), size=N_REP, replace=False, p=WeightProb)

                # K_set = np.argsort(Onecode_pred)[-N_REP:]
                N = np.ones(len(K_set))
                index_candidate, pred_set_list = self.word_paraphrase_nocombine(new_words, orig_prob_new, K_set,
                                                                                category_set, y)

                ucb_loop = 0
                INDEX = []
                while robust_flag == 1 and ucb_loop <= self.opt.ucbloop:
                    ucb_loop = ucb_loop + 1
                    iteration += 1
                    # print('K_set', K_set, file=log_f, flush=True)

                    ucb = self.UCBV(iteration, pred_set_list, N)
                    topk_feature_index = np.argsort(ucb)[-1]
                    INDEX.append(topk_feature_index)

                    Feat_max = K_set[topk_feature_index]
                    cand_max, pred_max = self.word_paraphrase_nocombine(new_words, orig_prob_new, [Feat_max],
                                                                        category_set, y)

                    new_words = self.New_Word(new_words, [(Feat_max, cand_max[0])], list_closest_neighbors)

                    arm_chain.append((Feat_max, cand_max[0]))
                    arm_pred.append(pred_max)

                    n_add = np.eye(len(N))[topk_feature_index]
                    N += n_add

                    pred_set_list_add = np.zeros(len(K_set))
                    pred_set_list_add[topk_feature_index] = pred_max
                    pred_set_list = pred_set_list + pred_set_list_add

                    time_end = time.time()
                    time_Dur = time_end - start_random
                    print('arm_chain', arm_chain, file=log_f, flush=True)
                    print('arm_pred', arm_pred, file=log_f, flush=True)
                    if arm_pred[-1] > TAU:
                        success.append(1)
                        num_armchain.append(len(arm_chain))
                        n_change.append(len(Counter(arm_chain).keys()))
                        time_success.append(time_Dur)
                        arm_chains.append(arm_chain)
                        arm_preds.append(arm_pred)
                        robust_flag = 0
                        break
                    if time_Dur > SECONDS:
                        print('The time is over', time_Dur, file=log_f, flush=True)
                        break
                if time_Dur > SECONDS:
                    print('The time is over', time_Dur, file=log_f, flush=True)
                    break

        return success, RN, num_armchain, time_success, arm_chains, arm_preds, n_change, NoAttack

def make_dir(path):
    if os.path.isdir(path):
        pass
    else:
        os.mkdir(path)

def main():
    opt = parse_args()
    Data_Type = 'Normal'
    UCB_LOOP = opt.ucbloop
    alpha = opt.alpha
    make_dir('./Logs/')
    make_dir('./Logs/ucb_rg/')
    log_f = open(
        './Logs/ucb_rg/Budget_%s_ALPHA=%s_N_REPLACE=%s_Time=%s_Loop=%s.bak' % (
            str(Budget), str(alpha), str(N_REPLACE), str(SECONDS), str(UCB_LOOP)),
        'w+')

    TITLE = '=== ' + 'UCB_RG' + ' target prob = ' + str(TAU) + ' changeRange = ' + str(N_REPLACE) + '_Time' + str(
        SECONDS) + ' ==='

    Model = {
        'Normal': './IPS_LSTM.par',
        'adversarial': './Mal_adv.705',
        'nuclear': './Mal_nuclear.722',
        'RS': 'Mal_RS.809',
        'Resnet': 'Mal_Resnet.170',
        'risk': './IPS_LSTM.par'
    }
    best_parameters_file = Model[Data_Type]

    print(TITLE)
    print(TITLE, file=log_f, flush=True)
    X = pickle.load(open(opt.test_path, 'rb'))
    label_file = './mal_test_label.pickle'
    y = pickle.load(open(label_file, 'rb'))
    attacker = Attacker(best_parameters_file, opt)

    success_num = 0
    success_data = []
    danger_data = []
    sample_index = []
    success_label = []
    arm_chains_samples = []
    arm_preds_samples = []
    NoAttack_num = 0
    F = []
    g = []
    F_V = []
    Total_iteration = 0
    Total_change = 0
    Total_time = 0
    i = 0

    for count, doc in enumerate(X):
        # if count !=14:
        #     continue
        # logging.info("Processing %d/%d documents", count + 1, len(X))
        print("====Sample %d/%d ======", count + 1, len(X), file=log_f, flush=True)
        success, RN, num_armchain, time_success, arm_chains, arm_preds, n_change, NoAttack_num = attacker.attack(count, doc, y[count],
                                                                                                   NoAttack_num, log_f,
                                                                                                   N_REPLACE,
                                                                                                   SubK_ratio)

        arm_chains_sample = []
        arm_preds_sample = 0
        if np.sum(success) >= 1:
            # print('all random success attack in this sample')
            SR_sample = np.sum(success) / RN
            success_num += SR_sample
            AI_sample = np.average(num_armchain)
            Achange = np.average(n_change)
            AT_sample = np.average(time_success)

            arm_chains_sample = arm_chains
            arm_preds_sample = arm_preds

            # sample_index.append(i)
            # success_label.append(ori_label)
            # for d in arm_chains_sample:
            #     success_sample.append(SetInsert(ori_data, d))
            #     danger_sample.append(SetInsert(ori_data, d[:-1]))

            Total_iteration += AI_sample
            Total_change += Achange
            Total_time += AT_sample
            # success_data.append(success_sample)
            # danger_data.append(danger_sample)

            print("  Number of iterations for this: ", AI_sample, file=log_f, flush=True)
            print(" Time: ", AT_sample, file=log_f, flush=True)

        print("--- Total Success Number: " + str(success_num) + " ---", file=log_f, flush=True)
        print("--- Total No Attack Number: " + str(NoAttack_num) + " ---", file=log_f, flush=True)
        if (count + 1 - NoAttack_num) != 0 and success_num != 0:
            print("--- success Ratio: " + str(success_num / (count + 1 - NoAttack_num)) + " ---", file=log_f,
                  flush=True)
            print("--- Mean Iteration: " + str(Total_iteration / success_num) + " ---", file=log_f,
                  flush=True)
            print("--- Mean Change: " + str(Total_change / success_num) + " ---", file=log_f,
                  flush=True)
            print("--- Mean Time: " + str(Total_time / success_num) + " ---", file=log_f,
                  flush=True)


if __name__ == '__main__':
    main()
