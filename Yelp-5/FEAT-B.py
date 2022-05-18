from __future__ import print_function
import logging
import argparse
import math
from copy import copy
import torch.nn as nn
import numpy as np
import time
from lm import NGramLangModel
from util import *
import spacy
import torch.nn.functional as F
import random
import pickle

# nlp = spacy.load('en', create_pipeline=wmd.WMD.create_spacy_pipeline)
nlp = spacy.load("en_core_web_lg")
NGRAM = 3
TAU = 0.8
N_NEIGHBOR = 15
N_REPLACE = 200  # len(doc)//N_REPLACE
SubK_ratio = 0.3
SECONDS = 360
Budget = 15
from gensim.models import KeyedVectors

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
    parser.add_argument('--word_delta', default=2, type=int, help='percentage of allowed word paraphasing')
    parser.add_argument('--model', default='LSTM', type=str, help='model: either CNN or LSTM')
    parser.add_argument('--train_path', action='store', default='./data/YelpFull/train.tsv', type=str,
                        dest='train_path',
                        help='Path to train data')
    parser.add_argument('--test_path', action='store', default='./data/YelpFull/Test_PredTrue_lr0.0001.tsv', type=str,
                        dest='test_path',
                        help='Path to test.txt data')
    parser.add_argument('--output_path', default='./data/changed_lstm', type=str,
                        help='Path to output changed test.txt data')
    parser.add_argument('--embedding_path', default='./data/paragram_300_sl999/paragram_300_sl999.txt', action='store',
                        dest='embedding_path',
                        help='Path to pre-trained embedding data')
    parser.add_argument('--model_path', action='store', default='./model/yelpfull/train_lr_0.000100_b=16_lstm_{17}.bak',
                        dest='model_path',
                        help='Path to pre-trained classifier model')
    parser.add_argument('--max_size', default=20000, type=int,
                        help='max amount of transformations to be processed by each iteration')
    parser.add_argument('--first_label', default='FAKE', help='The name of the first label that the model sees in the \
                         training data. The model will automatically set it to be the positive label. \
                         For instance, in the fake news dataset, the first label is FAKE.')

    return parser.parse_args()


class CNN(nn.Module):
    def __init__(self, sentence_len=200, kernel_sizes=[3, 4, 5], num_filters=100, embedding_dim=300,
                 pretrained_embeddings=None):
        super(CNN, self).__init__()
        self.sentence_len = sentence_len
        use_cuda = torch.cuda.is_available()
        self.kernel_sizes = kernel_sizes
        vocab_size = len(pretrained_embeddings)
        print(vocab_size)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(pretrained_embeddings)
        self.embedding.weight.requires_grad = False  # mode=="nonstatic"
        if use_cuda:
            self.embedding = self.embedding.to(device)
        conv_blocks = []
        for kernel_size in kernel_sizes:
            # maxpool kernel_size must <= sentence_len - kernel_size+1, otherwise, it could output empty
            maxpool_kernel_size = sentence_len - kernel_size + 1
            conv1d = nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=kernel_size * embedding_dim,
                               stride=embedding_dim)

            component = nn.Sequential(
                conv1d,
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=maxpool_kernel_size)
            )
            if use_cuda:
                component = component.to(device)

            conv_blocks.append(component)
        self.conv_blocks = nn.ModuleList(conv_blocks)  # ModuleList is needed for registering parameters in conv_blocks
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 2)

    def forward(self, x):  # x: (batch, sentence_len)
        x = self.embedding(x)  # embedded x: (batch, sentence_len, embedding_dim)
        #    input:  (batch, in_channel=embedding_dim, in_length=sentence_len),
        #    output: (batch, out_channel=num_filters, out_length=sentence_len-...)
        x = x.view(x.size(0), 1, -1)  # needs to convert x to (batch, 1, sentence_len*embedding_dim)
        x_list = [conv_block(x) for conv_block in self.conv_blocks]
        out = torch.cat(x_list, 2)
        out = out.view(out.size(0), -1)
        feature_extracted = out
        return F.softmax(self.fc(out), dim=1), feature_extracted


class Attacker(object):
    ''' main part of the attack model '''

    def __init__(self, X, opt):
        self.opt = opt
        self.suffix = 'wordonly-' + str(opt.word_delta)
        self.DELTA_W = int(opt.word_delta) * 0.1
        self.TAU_2 = 2
        self.TAU_wmd_s = 0.75
        self.TAU_wmd_w = 0.75
        # want do sentence level paraphrase first
        X = [doc.split() for doc in X]
        logging.info("Initializing language model...")
        print("Initializing language model...")
        self.lm = NGramLangModel(X, NGRAM)
        logging.info("Initializing word vectors...")
        print("Initializing word vectors...")
        self.w2v = KeyedVectors.load_word2vec_format(opt.embedding_path, encoding='utf-8', unicode_errors='ignore')
        logging.info("Loading pre-trained classifier...")
        print("Loading pre-trained classifier...")
        self.model = torch.load(opt.model_path, map_location=lambda storage, loc: storage)
        if torch.cuda.is_available():
            self.model.to(device)
        logging.info("Initializing vocabularies...")
        print("Initializing vocabularies...")
        self.src_vocab, self.label_vocab = self.load_vocab(opt.train_path)
        # to compute the gradient, we need to set up the optimizer first
        self.criterion = nn.CrossEntropyLoss().to(device)

    def word_paraphrase(self, words, poses, list_neighbors, y):
        candidates = [words]
        j = 1
        if self.opt.model == 'LSTM':
            max_size = int(self.opt.max_size) // len(words)
        else:
            max_size = int(self.opt.max_size) // self.model.sentence_len
        for pos in poses:
            closest_neighbors = list_neighbors[pos]
            if not closest_neighbors:
                j += 1
                continue
            current_candidates = copy(candidates)
            for repl in closest_neighbors:
                for c in candidates:
                    if len(current_candidates) > max_size:
                        break
                    corrupted = copy(c)
                    corrupted[pos] = repl
                    current_candidates.append(corrupted)
            candidates = copy(current_candidates)
            if len(candidates) > max_size:
                break
            j += 1

        if candidates:
            if self.opt.model == 'LSTM':
                candidate_var = text_to_var(candidates, self.src_vocab)
                pred_probs = self.model(candidate_var)

                pred_probs = torch.exp(pred_probs)
                # print('pred_probs', pred_probs)
                if pred_probs.shape[1] == 2:
                    pred_prob, best_candidate_id = pred_probs[:, 1 - y].max(dim=0)
                else:
                    best_candidate_id = pred_probs[:, y].min(dim=0)[1]
                    pred_probs[:, y] = 0
                    pred_prob, pred = pred_probs[best_candidate_id, :].max(dim=0)

                new_words = candidates[best_candidate_id.data]

            elif self.opt.model == 'CNN':
                candidate_var = self.text_to_var_CNN(candidates, self.src_vocab)
                pred_probs, _ = self.model(candidate_var)
                log_pred_prob, best_candidate_id = pred_probs[:, 1 - y].max(dim=0)
                new_words = candidates[best_candidate_id.data[0]]
                pred_prob = log_pred_prob.data[0]
        else:
            print('empty candidates!')
        return new_words, pred_prob, j, pred

    def word_paraphrase_nocombine(self, new_words, orig_prob_new, orig_pred_new, poses, list_closest_neighbors, y):
        index_candidate = np.zeros(len(poses), dtype=int)
        pred_candidate = np.zeros(len(poses))
        pred_label = -100 + np.zeros(len(poses))

        for i, candi_sample_index in enumerate(poses):
            candidates = [new_words]
            for neibor_index, neibor in enumerate(
                    list_closest_neighbors[candi_sample_index]):  # it omit the [] empty list

                corrupted = copy(new_words)
                candidate_var = text_to_var([corrupted], self.src_vocab)
                pred_probs_ori = self.model(candidate_var)
                # print('pred_probs',pred_probs.shape,pred_probs)
                pred_probs_ori = torch.exp(pred_probs_ori)

                corrupted[candi_sample_index] = neibor
                candidates.append(corrupted)

            if len(candidates) != 1:
                candidate_var = text_to_var(candidates, self.src_vocab)
                pred_probs = self.model(candidate_var)
                # print('pred_probs',pred_probs.shape,pred_probs)
                pred_probs = torch.exp(pred_probs)
                # print('pred_probs', len(pred_probs))
                if pred_probs.shape[1] == 2:
                    pred_prob, best_candidate_id = pred_probs[:, 1 - y].max(dim=0)
                else:
                    a = pred_probs - pred_probs[:, y].reshape([len(candidates), 1])
                    a[:, y] = -1000
                    best_candidate_id, pred = np.unravel_index(np.argmax(a.data.cpu(), axis=None), a.shape)
                    pred_prob = pred_probs[best_candidate_id, pred]
                    # best_candidate_id = pred_probs[:, y].min(dim=0)[1]
                    # pred_probs[:, y] = 0
                    # pred_prob, pred = pred_probs[best_candidate_id, :].max(dim=0)

                index_candidate[i] = best_candidate_id
                pred_candidate[i] = pred_prob
                pred_label[i] = pred
        pred_label[np.where(pred_label == -100)[0]] = orig_pred_new.data.cpu()
        # pred_candidate[np.where(pred_label == -100)[0]] = orig_prob_new
        return index_candidate, pred_candidate, pred_label

    def hidden(self, hidden_dim):
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(1, 1, hidden_dim).to(device))
            c0 = Variable(torch.zeros(1, 1, hidden_dim).to(device))
        else:
            h0 = Variable(torch.zeros(1, 1, hidden_dim))
            c0 = Variable(torch.zeros(1, 1, hidden_dim))
        return (h0, c0)

    def forward_lstm(self, embed,
                     model):  # copying the structure of LSTMClassifer, just omitting the first embedding layer
        lstm_out, hidden0 = model.rnn(embed, self.hidden(512))
        y = model.linear(lstm_out[-1])
        return y

    def forward_cnn(self, embed, model):
        x_list = [conv_block(embed) for conv_block in model.conv_blocks]
        out = torch.cat(x_list, 2)
        out = out.view(out.size(0), -1)
        return F.softmax(model.fc(out), dim=1)

    def text_to_var_CNN(self, docs, vocab):
        tensor = []
        max_len = self.model.sentence_len
        for doc in docs:
            vec = []
            for tok in doc:
                vec.append(vocab.stoi[tok])
            if len(doc) < max_len:
                vec += [0] * (max_len - len(doc))
            else:
                vec = vec[:max_len]
            tensor.append(vec)
        var = Variable(torch.LongTensor(tensor))
        if torch.cuda.is_available():
            var = var.to(device)
        return var

    def sentence_paraphrase(self, y, sentences, changed_pos, list_closest_neighbors):
        candidates = []
        responding_pos = []  # the index of the changed sentence
        for i, sentence in enumerate(sentences):
            if i in changed_pos:
                continue
            j = 0
            for p in list_closest_neighbors[i]:
                new_sentence = copy(sentences)
                new_sentence[i] = p
                new_sentence = (" ".join(new_sentence)).split()
                candidates.append(new_sentence)
                responding_pos.append((i, j))
                j += 1

        if candidates:
            m = len(candidates)
            if self.opt.model == 'LSTM':
                n = max([len(candidates[i]) for i in range(m)])
            else:
                n = self.model.sentence_len
            b = np.random.permutation(m)[:int(self.opt.max_size) // n]
            candidates = [candidates[i] for i in b]
            responding_pos = [responding_pos[i] for i in b]
            if self.opt.model == 'LSTM':
                candidate_var = text_to_var(candidates, self.src_vocab)
                pred_probs = self.model(candidate_var)
                log_pred_prob, best_candidate_id = pred_probs[:, 1 - y].max(dim=0)
                final_pos = responding_pos[best_candidate_id.data[0]][0]
                final_choice = responding_pos[best_candidate_id.data[0]][1]
                pred_prob = exp(log_pred_prob.data[0])
            else:
                candidate_var = self.text_to_var_CNN(candidates, self.src_vocab)
                pred_probs, _ = self.model(candidate_var)
                log_pred_prob, best_candidate_id = pred_probs[:, 1 - y].max(dim=0)
                final_pos = responding_pos[best_candidate_id.data[0]][0]
                final_choice = responding_pos[best_candidate_id.data[0]][1]
                pred_prob = log_pred_prob.data[0]
            print('final changed pos ' + str(final_pos) + ' from ' + sentences[final_pos] + ' ------->>>>> ' +
                  list_closest_neighbors[final_pos][final_choice] + ', score=' + str(pred_prob))
            sentences[final_pos] = list_closest_neighbors[final_pos][final_choice]
            return sentences, final_pos, pred_prob
        else:
            return sentences, -1, 0

    def load_vocab(self, path):
        src_field = data.Field()
        label_field = data.Field(pad_token=None, unk_token=None)
        dataset = data.TabularDataset(
            path=path, format='tsv',
            fields=[('text', src_field), ('label', label_field)]
        )
        src_field.build_vocab(dataset, max_size=100000, min_freq=2, vectors="glove.6B.300d")
        label_field.build_vocab(dataset)
        return src_field.vocab, label_field.vocab

    def Get_neigbor_list(self, words):
        list_closest_neighbors = []
        for pos, w in enumerate(words):
            if self.opt.model == 'CNN' and pos >= self.model.sentence_len: break
            try:
                closest_neighbors = self.w2v.most_similar(positive=[w.lower()],
                                                          topn=N_NEIGHBOR)  # get the 15 neighbors as the replacement set for this word.
            except:
                closest_neighbors = []
            closest_paraphrases = []
            closest_paraphrases.extend(closest_neighbors)
            # check if the words make sense
            valid_paraphrases = []
            doc1 = nlp(w)
            for repl, repl_sim in closest_paraphrases:
                doc2 = nlp(repl)  # ' '.join(repl_words))
                score = doc1.similarity(doc2)
                syntactic_diff = self.lm.log_prob_diff(words, pos, repl)
                logging.debug("Syntactic difference: %f", syntactic_diff)
                if score >= self.TAU_wmd_w and syntactic_diff <= self.TAU_2:  # check the chosen word useful or not
                    valid_paraphrases.append(repl)
            list_closest_neighbors.append(valid_paraphrases)  # closest_neighbors)
            if not closest_paraphrases:  # neighbors:
                # print('find no neighbor for word: '+w)
                continue

        return list_closest_neighbors

    def UCBV(self, round, pred_set_list, N):

        mean = np.mean(pred_set_list, axis=0)
        variation = np.var(pred_set_list, axis=0)
        delta = np.sqrt((8 * variation * math.log(round)) / (N)) + (8 * math.log(round) / (N))
        ucb = mean + delta

        return ucb

    def New_Word(self, words, arm_chain, list_closest_neighbors):
        new_word = copy(words)
        if len(arm_chain) == 0:
            pass
        else:
            for pos in arm_chain:
                if pos[1] != 0:
                    new_word[pos[0]] = list_closest_neighbors[pos[0]][pos[1] - 1]

        return new_word

    def attack(self, count, doc, y, NoAttack, log_f, N_REPLACE, SubK_ratio):
        st = time.time()
        # ---------------------------------word paraphrasing----------------------------------------------#
        words = doc.split()
        words_before = copy(words)
        best_words = copy(words)
        # check if the value of this doc to be right
        if self.opt.model == 'LSTM':
            doc_var = text_to_var([words], self.src_vocab)
        else:
            doc_var = self.text_to_var_CNN([words], self.src_vocab)
        orig_prob, orig_pred = classify(doc_var, self.model)
        pred, pred_prob = orig_pred, orig_prob
        if not (pred == y):  # attack success
            print(" this original samples predict wrong")
            NoAttack = NoAttack + 1
            print(pred, y)
            return words, 0, 0, [], 0, 0, 0, NoAttack
        best_score = 1 - pred_prob

        # now word level paraphrasing
        ## step1: find the neighbor for all the words.
        neigbor = self.Get_neigbor_list(words)

        changed_pos = set()
        iteration = 0
        recompute = True
        n_change = 0
        if self.opt.model == 'CNN':
            lword = min(len(words), self.model.sentence_len)
        else:
            lword = len(words)
        changed_words = []

        allCode_ori = list(range(len(words)))
        allCode = []
        for code_ind in allCode_ori:
            if len(neigbor[code_ind]) != 0:
                allCode.append(code_ind)

        if len(allCode) <= N_REPLACE:
            N_REP = len(allCode)
        else:
            N_REP = N_REPLACE

        RN = 5  # this is the time to repeat all the random process. to get a more stable result.
        success = []
        num_armchain = []
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

            arm_chain = []  # the candidate set is S after selecting code process
            arm_pred = []
            iteration = 0
            N = np.ones(len(K_set))
            time_Dur = 0
            robust_flag = 1
            new_words = copy(words)
            while robust_flag == 1 and len(arm_chain) <= Budget and time_Dur <= SECONDS:
                iteration += 1
                # new_neigbor = self.Get_neigbor_list(new_words)
                changed_words = []
                # for i in range(len(words)):
                #     if words_before[i] != new_words[i]:
                #         changed_words.append(words_before[i])
                # print('changed_words', (changed_words,new_words[i]), len(changed_words))
                new_words_var = text_to_var([new_words], self.src_vocab)
                orig_prob_new, orig_pred_new = classify(new_words_var, self.model)

                index_candidate, pred_candidate, pred_label = self.word_paraphrase_nocombine(new_words, orig_prob_new,
                                                                                             orig_pred_new, K_set,
                                                                                             neigbor, y)
                pred_set_list = pred_candidate
                ucb = self.UCBV(iteration, pred_set_list, N)

                topk_feature_index = np.argsort(ucb)[-1]
                pred_max = pred_candidate[topk_feature_index]
                Feat_max = K_set[topk_feature_index]
                cand_max = index_candidate[topk_feature_index]

                feature_n = 1
                while (Feat_max, cand_max) in arm_chain or (Feat_max, cand_max) == (Feat_max, 0):
                    if feature_n <= N_REP - 1:
                        topk_feature_index = np.argsort(ucb)[-(1 + feature_n)]
                        pred_max = pred_candidate[topk_feature_index]
                        Feat_max = K_set[topk_feature_index]
                        cand_max = index_candidate[topk_feature_index]
                        feature_n += 1
                    else:
                        break

                if (feature_n > N_REP - 1) and (
                        (Feat_max, cand_max) in arm_chain or (Feat_max, cand_max) == (Feat_max, 0)):
                    break

                new_words = self.New_Word(new_words, [(Feat_max, cand_max)], neigbor)

                n_add = np.eye(len(N))[topk_feature_index]
                N += n_add[0]
                arm_chain.append((Feat_max, cand_max))
                arm_pred.append(pred_max)
                time_end = time.time()
                time_Dur = time_end - start_random
                if pred_max > TAU:
                    success.append(1)
                    num_armchain.append(len(arm_chain))
                    time_success.append(time_Dur)
                    arm_chains.append(arm_chain)
                    arm_preds.append(arm_pred)
                    robust_flag = 0
                    # print('K_set', K_set, file=log_f, flush=True)
                    # print('arm_chain',arm_chain , file=log_f, flush=True)
                    # print('attack success', file=log_f, flush=True)
                    break

                if time_Dur > SECONDS:
                    print('The time is over', time_Dur, file=log_f, flush=True)
                    break

        return success, RN, num_armchain, time_success, arm_chains, arm_preds


def main():
    opt = parse_args()
    log_f = open(
        './Logs/UCB_R_Budget_%s_N=%s_TAU=%s_N_REPLACE=%s_Time=%s.bak' % (
        str(Budget), str(N_NEIGHBOR), str(TAU), str(N_REPLACE), str(SECONDS)),
        'w+')

    TITLE = '=== ' + 'UCB_R' + ' target prob = ' + str(TAU) + ' changeRange = ' + str(N_REPLACE) + '_Time' + str(
        SECONDS) + ' ==='

    print(TITLE)
    print(TITLE, file=log_f, flush=True)
    X_train, y_train = read_data_multi_class(opt.train_path)
    # X, y = read_data_multi_class(opt.test_path)
    [X, y] = pickle.load(open('./data/YelpFull/test_selected_balance.pkl', 'rb'))
    attacker = Attacker(X_train, opt)
    del X_train
    del y_train

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
    Total_time = 0
    i = 0
    suffix = 'wordonly-' + str(opt.word_delta)

    for count, doc in enumerate(X):
        # logging.info("Processing %d/%d documents", count + 1, len(X))
        print("====Sample %d/%d ======", count + 1, len(X), file=log_f, flush=True)
        success, RN, num_armchain, time_success, arm_chains, arm_preds = attacker.attack(count, doc, y[count],
                                                                                         NoAttack_num, log_f, N_REPLACE,
                                                                                         SubK_ratio)

        arm_chains_sample = []
        arm_preds_sample = 0
        if np.sum(success) >= 1:
            # print('all random success attack in this sample')
            SR_sample = np.sum(success) / RN
            success_num += SR_sample
            AI_sample = np.average(num_armchain)
            AT_sample = np.average(time_success)

            arm_chains_sample = arm_chains
            arm_preds_sample = arm_preds

            # sample_index.append(i)
            # success_label.append(ori_label)
            # for d in arm_chains_sample:
            #     success_sample.append(SetInsert(ori_data, d))
            #     danger_sample.append(SetInsert(ori_data, d[:-1]))

            Total_iteration += AI_sample
            Total_time += AT_sample
            # success_data.append(success_sample)
            # danger_data.append(danger_sample)

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
            print("--- Mean Time: " + str(Total_time / success_num) + " ---", file=log_f,
                  flush=True)


if __name__ == '__main__':
    main()
