from __future__ import print_function
import logging
import argparse
import math
from copy import copy
import time
import random
import pickle
from collections import Counter
import copy
from utils import *
from model import *

NGRAM = 3
TAU = 0.5

SubK_ratio = 0.3
SECONDS = 300
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
    parser.add_argument('--test_path', action='store', default='./dataset/mal_test_funccall.pickle', type=str,
                        dest='test_path',
                        help='Path to test.txt data')
    parser.add_argument('--max_size', default=20000, type=int,
                        help='max amount of transformations to be processed by each iteration')
    parser.add_argument('--nrep', default=20, type=int,
                        help='N_Replace')
    parser.add_argument('--alpha', default=8, type=int,
                        help='alpha')

    return parser.parse_args()


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


    def word_paraphrase_nocombine(self, new_words, orig_pred_new, poses, list_closest_neighbors, y):
        index_candidate = np.zeros(len(poses), dtype=int)
        pred_candidate = np.zeros(len(poses))

        for i, candi_sample_index in enumerate(poses):
            candidates = []
            best_pred_prob = 0.0
            neighbors = np.random.choice(list_closest_neighbors, 100)
            for neibor in neighbors:
                corrupted = copy.deepcopy(new_words)
                corrupted[candi_sample_index] = neibor
                candidates.append(corrupted)
            if len(candidates) != 1:
                batch_size = 32
                n_batches = int(np.ceil(float(len(candidates)) / float(batch_size)))
                for index in range(n_batches):  # n_batches

                    batch_diagnosis_codes = candidates[batch_size * index: batch_size * (index + 1)]
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
                    t_diagnosis_codes = torch.autograd.Variable(t_diagnosis_codes.data, requires_grad=True)

                    pred_probs = self.model(model_input, t_diagnosis_codes)
                    result = (1 - pred_probs[:, y]).max(dim=0)
                    candidate_id = result[1].cpu().numpy().item() + batch_size * index
                    pred_prob = result[0].cpu().detach().numpy().item()

                    if pred_prob > best_pred_prob:
                        best_pred_prob = pred_prob
                        index_candidate[i] = neighbors[candidate_id]
                        pred_candidate[i] = pred_prob

        return index_candidate, pred_candidate

    def UCBV(self, round, pred_set_list, N):

        mean = pred_set_list / N
        variation = (pred_set_list - mean) ** 2 / N
        delta = np.sqrt((self.opt.alpha * variation * math.log(round)) / N) + (self.opt.alpha * math.log(round) / N)
        ucb = mean + delta

        return ucb

    def New_Word(self, words, arm_chain, list_closest_neighbors):
        new_word = copy.deepcopy(words)
        if len(arm_chain) == 0:
            pass
        else:
            for pos in arm_chain:
                # new_word[pos[0]] = list_closest_neighbors[pos[0]][pos[1] - 1]
                new_word[pos[0]] = list_closest_neighbors[pos[1]]
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

    def attack(self, count, words, y, NoAttack, log_f, N_REPLACE, SubK_ratio):
        # check if the value of this doc to be right
        orig_prob, orig_pred = self.classify(words, y)
        pred, pred_prob = orig_pred, orig_prob
        if not (pred == y):  # attack success
            print(" this original samples predict wrong")
            NoAttack = NoAttack + 1
            print(pred, y)
            return 0, 0, [], 0, 0, 0, NoAttack
        best_score = 1 - pred_prob

        # now word level paraphrasing
        ## step1: find the neighbor for all the words.
        # list_closest_neighbors = self.Get_neigbor_list(words)
        list_closest_neighbors = range(1104)

        changed_pos = set()
        iteration = 0
        recompute = True
        n_change = 0
        lword = 20
        changed_words = []

        allCode = range(lword)


        if len(allCode) <= N_REPLACE:
            N_REP = len(allCode)
        else:
            N_REP = N_REPLACE

        RN = 1  # this is the time to repeat all the random process. to get a more stable result.
        success = []
        num_armchain = []
        n_change = []
        time_success = []
        arm_chains = []
        arm_preds = [pred_prob]

        for n in range(RN):
            # print("random index :",n)
            start_random = time.time()
            K_set = random.sample(allCode, N_REP)
            # performance index parameter
            # 1 success = [0,1,1,1,1]
            # 2 code number with success = [2,3,4,5,6]
            # 3 time computation with success = [2,3,4,5,6]
            # 4 search ability = [code number with success] /[time computation with success]
            # print('K_set', K_set, file=log_f, flush=True)
            arm_chain = []  # the candidate set is S after selecting code process
            arm_pred = []
            iteration = 0
            N = np.ones(len(K_set))
            # cand_visited = {}
            # for i in range(N_REP):
            #     cand_visited[i] = []
            time_Dur = 0
            robust_flag = 1
            new_words = copy.deepcopy(words)
            orig_prob_new, orig_pred_new = self.classify(new_words, y)
            index_candidate, pred_set_list = self.word_paraphrase_nocombine(new_words, orig_pred_new, K_set,
                                                                            list_closest_neighbors, y)
            print('index_candidate', index_candidate, pred_set_list, file=log_f, flush=True)
            INDEX = []
            while robust_flag == 1 and len(Counter(arm_chain).keys()) <= Budget and time_Dur <= SECONDS:
                iteration += 1

                ucb = self.UCBV(iteration, pred_set_list, N)
                topk_feature_index = np.argsort(ucb)[-1]
                INDEX.append(topk_feature_index)

                Feat_max = K_set[topk_feature_index]
                cand_max, pred_max = self.word_paraphrase_nocombine(new_words, orig_pred_new, [Feat_max],
                                                                    list_closest_neighbors, y)
                print('cand_max, pred_max', (cand_max, pred_max), file=log_f, flush=True)

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
                # print('attack success', file=log_f, flush=True)
                if arm_pred[-1] > TAU:
                    success.append(1)
                    num_armchain.append(len(arm_chain))
                    n_change.append(len(Counter(arm_chain).keys()))
                    time_success.append(time_Dur)
                    arm_chains.append(arm_chain)
                    arm_preds.append(arm_pred)
                    robust_flag = 0

                    # print('arm_chain', arm_chain, file=log_f, flush=True)
                    # print('attack success', file=log_f, flush=True)
                    break
                if time_Dur > SECONDS:
                    print('The time is over', time_Dur, file=log_f, flush=True)
                    break
            if time_Dur > SECONDS:
                print('The time is over', time_Dur, file=log_f, flush=True)
                break

        return success, RN, num_armchain, time_success, arm_chains, arm_preds, n_change


def main():
    opt = parse_args()
    Data_Type = 'Normal'
    N_REPLACE = opt.nrep
    alpha = opt.alpha
    make_dir('./Logs/')
    make_dir('./Logs/FEAT/')
    log_f = open(
        './Logs/FEAT/Budget_%s_ALPHA=%s_N_REPLACE=%s_Time=%s.bak' % (
            str(Budget), str(alpha), str(N_REPLACE), str(SECONDS)),
        'w+')

    TITLE = '=== ' + 'UCB_R' + ' target prob = ' + str(TAU) + ' changeRange = ' + str(N_REPLACE) + '_Time' + str(
        SECONDS) + ' ==='

    Model = {
        'Normal': './IPS_LSTM.par',
    }
    best_parameters_file = Model[Data_Type]

    print(TITLE)
    print(TITLE, file=log_f, flush=True)
    X = pickle.load(open(opt.test_path, 'rb'))
    label_file = './dataset/mal_test_label.pickle'
    y = pickle.load(open(label_file, 'rb'))
    attacker = Attacker(best_parameters_file, opt)

    success_num = 0
    NoAttack_num = 0
    Total_iteration = 0
    Total_change = 0
    Total_time = 0

    for count, doc in enumerate(X):
        # logging.info("Processing %d/%d documents", count + 1, len(X))
        print("====Sample %d/%d ======", count + 1, len(X), file=log_f, flush=True)
        success, RN, num_armchain, time_success, arm_chains, arm_preds, n_change = attacker.attack(count, doc, y[count],
                                                                                                   NoAttack_num, log_f,
                                                                                                   N_REPLACE,
                                                                                                   SubK_ratio)

        if np.sum(success) >= 1:
            # print('all random success attack in this sample')
            SR_sample = np.sum(success) / RN
            success_num += SR_sample
            AI_sample = np.average(num_armchain)
            Achange = np.average(n_change)
            AT_sample = np.average(time_success)

            Total_iteration += AI_sample
            Total_change += Achange
            Total_time += AT_sample

            print("  Number of iterations for this: ", AI_sample, file=log_f, flush=True)
            print(" Time: ", AT_sample, file=log_f, flush=True)

        print("* SUCCESS Number NOW: %d " % (success_num), file=log_f, flush=True)
        print("* NoAttack Number NOW: %d " % (NoAttack_num), file=log_f, flush=True)

        print("--- Total Success Number: " + str(success_num) + " ---", file=log_f, flush=True)
        print("--- Total No Attack Number: " + str(NoAttack_num) + " ---", file=log_f, flush=True)
        if (count - NoAttack_num) != 0 and success_num != 0:
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
