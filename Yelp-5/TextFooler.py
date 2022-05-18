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
from collections import Counter
import criteria
from nltk import pos_tag
import nltk
nltk.download('punkt')
nltk.download('maxent_treebank_pos_tagger')

# nlp = spacy.load('en', create_pipeline=wmd.WMD.create_spacy_pipeline)
nlp = spacy.load("en_core_web_lg")
NGRAM = 3
TAU = 0.5
N_NEIGHBOR = 20
N_REPLACE = 20 # len(doc)//N_REPLACE
SubK_ratio = 0.5
SECONDS = 1000
Budget = 6
ucb_num = 4
ALPHA = 0.5
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


    ## embedding for preparing the similarity
    parser.add_argument("--word_embeddings_path",
                        type=str,
                        default='src/.vector_cache/glove.6B.300d.txt',
                        help="path to the word embeddings for the target model")
    parser.add_argument("--counter_fitting_embeddings_path",
                        type=str,
                        default= 'src/counter-fitted-vectors.txt',
                        # required=True,
                        help="path to the counter-fitting embeddings we used to find synonyms")
    parser.add_argument("--counter_fitting_cos_sim_path",
                        type=str,
                        default='src/cos_sim_counter_fitting.npy',
                        help="pre-compute the cosine similarity scores based on the counter-fitting embeddings")
    parser.add_argument("--USE_cache_path",
                        type=str,
                        default= '',
                        # required=True,
                        help="Path to the USE encoder cache.")
    parser.add_argument("--output_dir",
                        type=str,
                        default='adv_results',
                        help="The output directory where the attack results will be written.")

    ## Model hyperparameters
    parser.add_argument("--sim_score_window",
                        default=15,
                        type=int,
                        help="Text length or token number to compute the semantic similarity score")
    parser.add_argument("--import_score_threshold",
                        default=-1.,
                        type=float,
                        help="Required mininum importance score.")
    parser.add_argument("--sim_score_threshold",
                        default=0.7,
                        type=float,
                        help="Required minimum semantic similarity score.")
    parser.add_argument("--synonym_num",
                        default=50,
                        type=int,
                        help="Number of synonyms to extract")
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Batch size to get prediction")
    parser.add_argument("--data_size",
                        default=1000,
                        type=int,
                        help="Data size to create adversaries")
    parser.add_argument("--perturb_ratio",
                        default=0.,
                        type=float,
                        help="Whether use random perturbation for ablation study")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="max sequence length for BERT target model")



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

    def __init__(self, X, args):
        self.args = args
        self.suffix = 'wordonly-' + str(args.word_delta)
        self.DELTA_W = int(args.word_delta) * 0.1
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
        self.w2v = KeyedVectors.load_word2vec_format(args.embedding_path, encoding='utf-8', unicode_errors='ignore')
        logging.info("Loading pre-trained classifier...")
        print("Loading pre-trained classifier...")
        self.model = torch.load(args.model_path, map_location=lambda storage, loc: storage)
        if torch.cuda.is_available():
            self.model.to(device)
        logging.info("Initializing vocabularies...")
        print("Initializing vocabularies...")
        self.src_vocab, self.label_vocab = self.load_vocab(args.train_path)
        # to compute the gradient, we need to set up the argsimizer first
        self.criterion = nn.CrossEntropyLoss().to(device)

    def word_paraphrase(self, words, poses, list_neighbors, y):
        candidates = [words]
        j = 1
        if self.args.model == 'LSTM':
            max_size = int(self.args.max_size) // len(words)
        else:
            max_size = int(self.args.max_size) // self.model.sentence_len
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
            if self.args.model == 'LSTM':
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

            elif self.args.model == 'CNN':
                candidate_var = self.text_to_var_CNN(candidates, self.src_vocab)
                pred_probs, _ = self.model(candidate_var)
                log_pred_prob, best_candidate_id = pred_probs[:, 1 - y].max(dim=0)
                new_words = candidates[best_candidate_id.data[0]]
                pred_prob = log_pred_prob.data[0]
        else:
            print('empty candidates!')
        return new_words, pred_prob, j, pred

    def word_paraphrase_nocombine(self, new_words, orig_pred_new, poses, list_closest_neighbors, y):
        index_candidate = np.zeros(len(poses), dtype=int)
        pred_candidate = np.zeros(len(poses))
        for i, candi_sample_index in enumerate(poses):
            candidates = [new_words]
            for neibor_index, neibor in enumerate(
                    list_closest_neighbors[candi_sample_index]):  # it omit the [] empty list

                corrupted = copy(new_words)
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
                    result = (1 - pred_probs[:, y]).max(dim=0)
                    best_candidate_id = result[1].cpu().numpy().item()
                    pred_prob = result[0].cpu().detach().numpy().item()

                index_candidate[i] = best_candidate_id
                pred_candidate[i] = pred_prob

        return index_candidate, pred_candidate

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
            if self.args.model == 'LSTM':
                n = max([len(candidates[i]) for i in range(m)])
            else:
                n = self.model.sentence_len
            b = np.random.permutation(m)[:int(self.args.max_size) // n]
            candidates = [candidates[i] for i in b]
            responding_pos = [responding_pos[i] for i in b]
            if self.args.model == 'LSTM':
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

    def UCBV(self, round, pred_set_list, N,alpha):

        mean = pred_set_list/N
        variation = (pred_set_list - mean)**2/N
        delta = np.sqrt((alpha * variation * math.log(round)) / (N)) + (alpha * math.log(round) / (N))
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
    def Pred_new(self,new_words,y):
        candidate_var = text_to_var([new_words], self.src_vocab)
        pred_probs = self.model(candidate_var)
        # print('pred_probs',pred_probs.shape,pred_probs)
        pred_probs = torch.exp(pred_probs)
        # print('pred_probs', len(pred_probs))
        if pred_probs.shape[1] == 2:
            pred_prob, best_candidate_id = pred_probs[:, 1 - y].max(dim=0)
        else:
            result = (1 - pred_probs[:, y]).max(dim=0)
            pred_prob = result[0].cpu().detach().numpy().item()

        return pred_prob
    def Grad_index(self, words, list_closest_neighbors, changed_pos, y):
        doc_var = text_to_var([words], self.src_vocab)
        embed_doc = self.model.embedding(doc_var)
        embed_doc = Variable(embed_doc.data,
                             requires_grad=True)  # make it a leaf node and requires gradient
        output = self.forward_lstm(embed_doc, self.model)

        if torch.cuda.is_available():
            loss = self.criterion(output, Variable(torch.LongTensor([y])).to(device))
        else:
            loss = self.criterion(output, Variable(torch.LongTensor([y])))
        loss.backward()

        score = np.zeros(len(words))  # ,1+N_NEIGHBOR*2))
        # save the score of the nearest paraphrases and the original word
        for pos, w in enumerate(words):
            # if pos in changed_pos or not list_closest_neighbors[pos]:
            #     continue  # don't want to change again, or if there's no choice of replacement
            a = embed_doc.grad.data[pos, 0, :].view(300)  # score based on the gradient of each word
            score[pos] = torch.dot(a, a)

        # min_score = []  # select the high scores
        # valid_n = 0
        # for i in range(len(list_closest_neighbors)):
        #     if list_closest_neighbors[i] and not i in changed_pos:
        #         min_score.append(-score[i])
        #         valid_n += 1
        #     else:
        #         min_score.append(10000)

        return score

    def pick_most_similar_words_batch(self,src_words, sim_mat, idx2word, ret_count=10, threshold=0.):
        """
        embeddings is a matrix with (d, vocab_size)
        """
        sim_order = np.argsort(-sim_mat[src_words, :])[:, 1:1 + ret_count]
        sim_words, sim_values = [], []
        for idx, src_word in enumerate(src_words):
            sim_value = sim_mat[src_word][sim_order[idx]]
            mask = sim_value >= threshold
            sim_word, sim_value = sim_order[idx][mask], sim_value[mask]
            sim_word = [idx2word[id] for id in sim_word]
            sim_words.append(sim_word)
            sim_values.append(sim_value)
        return sim_words, sim_values

    def predictor(self,new_texts):

        probs_all = []
        for text in new_texts:
            words_var = text_to_var([text], self.src_vocab)
            orig_prob, orig_pred, orig_prob_all = classify_all(words_var, self.model)
            probs_all.append(orig_prob_all.detach().cpu().numpy())

        return torch.as_tensor(probs_all).cuda()

    def attack(self, count, doc, y, NoAttack, log_f, N_REPLACE, SubK_ratio,stop_words_set,
                                            word2idx, idx2word, cos_sim,sim_predictor,
                                            sim_score_threshold,
                                            import_score_threshold,
                                            sim_score_window,
                                            synonym_num
                                            ):
        # ---------------------------------word paraphrasing----------------------------------------------#
        words = doc.split()

        # check if the value of this doc to be right
        if self.args.model == 'LSTM':
            doc_var = text_to_var([words], self.src_vocab)
        else:
            doc_var = self.text_to_var_CNN([words], self.src_vocab)
        orig_prob, orig_pred,pred_all = classify_all(doc_var, self.model)
        pred, pred_prob = orig_pred, orig_prob

        text_ls = words
        len_text = len(text_ls)
        if not (pred == y):  # attack success
            print(" this original samples predict wrong")
            NoAttack = NoAttack + 1
            print(pred, y)
            return words, 0, y, y,pred_prob, 0, 0, 0,NoAttack

        else:
            len_text = len(text_ls)
            if len_text < sim_score_window:
                sim_score_threshold = 0.1  # shut down the similarity thresholding function
            half_sim_score_window = (sim_score_window - 1) // 2
            num_queries = 1


        # get the pos and verb tense info
        pos_ls = pos_tag(text_ls)


        ## Line 2-4: compute the importance score for each word
        I_w = []
        for pos,word in enumerate(words):
            words_now = copy(words)
            words_now[pos] = ''
            words_now.remove('')

            doc_var = text_to_var([words_now], self.src_vocab)
            new_prob, new_pred,new_prob_y = classify_label(doc_var, self.model,y)

            if new_pred == y:
                score_word = orig_prob - new_prob_y
            else:
                score_word = (orig_prob - new_prob_y) + (new_prob - pred_all[new_pred])

            I_w.append(score_word)

        ## Line 6-7
        # import_scores = np.argsort(np.array(I_w))
        # W_ordered = []
        # for i_order in score_order:
        #     W_ordered.append(words[i_order])


        # get importance score
        leave_1_texts = [text_ls[:ii] + ['<oov>'] + text_ls[min(ii + 1, len_text):] for ii in range(len_text)]

        num_queries += len(leave_1_texts)

        import_scores = np.array(I_w)
        # get words to perturb ranked by importance score for word in words_perturb
        words_perturb = []
        for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True):
            try:
                if score > import_score_threshold and text_ls[idx] not in stop_words_set:
                    words_perturb.append((idx, text_ls[idx]))
            except:
                print(idx, len(text_ls), import_scores.shape, text_ls, len(leave_1_texts))

        # find synonyms
        words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
        synonym_words, _ = self.pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, ALPHA)
        synonyms_all = []
        for idx, word in words_perturb:
            if word in word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))


        # start replacing and attacking
        robust_flag = 1
        text_prime = text_ls[:]
        text_cache = text_prime[:]
        num_perturbed = 0
        orig_label = orig_pred
        st = time.time()
        changed_words = []
        iteration = 0
        for idx, synonyms in synonyms_all:
            iteration = iteration +1
            new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]

            new_probs = self.predictor(new_texts)

            # compute semantic similarity
            if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = idx - half_sim_score_window
                text_range_max = idx + half_sim_score_window + 1
            elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = 0
                text_range_max = sim_score_window
            elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
                text_range_min = len_text - sim_score_window
                text_range_max = len_text
            else:
                text_range_min = 0
                text_range_max = len_text
            semantic_sims = \
            sim_predictor.semantic_sim([' '.join(text_cache[text_range_min:text_range_max])] * len(new_texts),
                                       list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[0]

            num_queries += len(new_texts)
            if len(new_probs.shape) < 2:
                new_probs = new_probs.unsqueeze(0)
            new_probs_mask = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
            # prevent bad synonyms
            new_probs_mask *= (semantic_sims >= sim_score_threshold)
            # # prevent incompatible pos
            # synonyms_pos_ls = [pos_tag(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
            #                    if len(new_text) > 10 else pos_tag(new_text)[idx] for new_text in new_texts]
            # pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
            # new_probs_mask *= pos_mask

            if np.sum(new_probs_mask) > 0:
                text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
                num_perturbed += 1
                changed_words.append(text_prime[idx])

                new_label = torch.argmax(self.predictor([text_prime]))
                new_prob = self.predictor([text_prime])[0][new_label]

                print('solution found', file=log_f, flush=True)
                Time = time.time() - st
                return ' '.join(
                    text_prime), num_perturbed, orig_label, new_label, new_prob, num_queries, Time, changed_words, NoAttack

            else:
                new_label_probs = new_probs[:, orig_label] \
                        #           + torch.from_numpy(
                        # (semantic_sims < sim_score_threshold) + (1 - pos_mask).astype(float)).float().cuda()
                new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
                if new_label_prob_min < orig_prob:
                    text_prime[idx] = synonyms[new_label_prob_argmin]
                    num_perturbed += 1
                    changed_words.append(text_prime[idx])

            text_cache = text_prime[:]

            new_label = torch.argmax(self.predictor([text_prime]))
            new_prob = self.predictor([text_prime])[0][new_label]

            ## decide the prediction is right or wrong?
            if new_label != y and new_prob >= TAU:
                robust_flag = 0
                print('solution found', file=log_f, flush=True)
                Time = time.time() - st
                return ' '.join(text_prime), num_perturbed, orig_label, new_label,new_prob, num_queries, Time,changed_words,NoAttack

            if iteration == len(synonyms_all):
                print(" having searched all words", file=log_f, flush=True)
                Time = time.time() - st
                return ' '.join(text_prime), num_perturbed, orig_label,new_label,new_prob, num_queries, Time,changed_words,NoAttack

            if robust_flag == 1 and num_perturbed <= Budget and time.time() - st <= SECONDS:
                continue
            else:
                print('Attack Fail found', file=log_f, flush=True)
                Time = time.time() - st
                return ' '.join(text_prime), num_perturbed, orig_label,new_label,new_prob, num_queries, Time,changed_words,NoAttack




import tensorflow as tf
import tensorflow_hub as hub
import os
# # unicode environ
# environ = _createenviron()
# del _createenviron
class USE(object):
    def __init__(self, cache_path):
        super(USE, self).__init__()
        os.environ['TFHUB_CACHE_DIR'] = cache_path
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
        self.embed = hub.Module(module_url)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.build_graph()
        self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def build_graph(self):
        self.sts_input1 = tf.placeholder(tf.string, shape=(None))
        self.sts_input2 = tf.placeholder(tf.string, shape=(None))

        sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
        sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
        self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
        self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

    def semantic_sim(self, sents1, sents2):
        scores = self.sess.run(
            [self.sim_scores],
            feed_dict={
                self.sts_input1: sents1,
                self.sts_input2: sents2,
            })
        return scores

def main():
    args = parse_args()

    log_f = open(
        './Logs/TEXTFooler_Budget=%s_ALPHA=%s_N=%s_TAU=%s_N_REPLACE=%s_Time=%s.bak' % (
        str(Budget),str(ALPHA), str(N_NEIGHBOR), str(TAU), str(N_REPLACE), str(SECONDS)),
        'w+')

    TITLE = '=== ' + 'TEXTFooler_Budget' + ' target prob = ' + str(TAU) + ' changeRange = ' + str(N_REPLACE) + '_Time' + str(
        SECONDS) + ' ==='

    print(TITLE)
    print(TITLE, file=log_f, flush=True)
    X_train, y_train = read_data_multi_class(args.train_path)
    # X, y = read_data_multi_class(args.test_path)
    [X, y] = pickle.load(open('./data/YelpFull/test_selected_long_balanced30_200.pkl', 'rb'))
    attacker = Attacker(X_train, args)
    del X_train
    del y_train


    # prepare synonym extractor
    # build dictionary via the embedding file
    idx2word = {}
    word2idx = {}

    print("Building vocab...")
    with open(args.counter_fitting_embeddings_path, 'r') as ifile:
        for line in ifile:
            word = line.split()[0]
            if word not in idx2word:
                idx2word[len(idx2word)] = word
                word2idx[word] = len(idx2word) - 1

    print("Building cos sim matrix...")
    if args.counter_fitting_cos_sim_path:
        # load pre-computed cosine similarity matrix if provided
        print('Load pre-computed cosine similarity matrix from {}'.format(args.counter_fitting_cos_sim_path))
        cos_sim = np.load(args.counter_fitting_cos_sim_path)
    else:
        # calculate the cosine similarity matrix
        print('Start computing the cosine similarity matrix!')
        embeddings = []
        with open(args.counter_fitting_embeddings_path, 'r') as ifile:
            for line in ifile:
                embedding = [float(num) for num in line.strip().split()[1:]]
                embeddings.append(embedding)
        embeddings = np.array(embeddings)
        product = np.dot(embeddings, embeddings.T)
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        cos_sim = product / np.dot(norm, norm.T)
    print("Cos sim import finished!")

    # build the semantic similarity module
    use = USE(' ')
    # get the stop words set
    stop_words_set = criteria.get_stopwords()

    suc = 0
    time = 0
    Num_change = 0
    Iter = 0
    Query = 0
    NoAttack = 0

    for count, doc in enumerate(X):
        # if count !=31:
        #     continue
        # logging.info("Processing %d/%d documents", count + 1, len(X))
        print("====Sample %d/%d ======", count + 1, len(X), file=log_f, flush=True)
        new_text, num_changed, orig_label, \
        new_label,new_prob, num_queries,Time,changed_words,NoAttack = attacker.attack(count, doc, y[count],
                                                                                         NoAttack, log_f, N_REPLACE,
                                                                                         SubK_ratio,stop_words_set,
                                            word2idx, idx2word, cos_sim,sim_predictor=use,
                                            sim_score_threshold=args.sim_score_threshold,
                                            import_score_threshold=args.import_score_threshold,
                                            sim_score_window=args.sim_score_window,
                                            synonym_num=args.synonym_num)

        print(" The cost Time", Time, file=log_f, flush=True)
        print(" The number changed", num_changed, file=log_f, flush=True)
        print(" The changed words", changed_words, file=log_f, flush=True)
        print(" NoAttack Number", NoAttack, file=log_f, flush=True)
        v = float(new_prob)
        if v >= TAU and orig_label != new_label:
            suc += 1
            time = time + Time
            Num_change = Num_change + num_changed
            Iter = Iter + num_changed
            Query = Query + num_queries

        else:
            print('Attack fail', file=log_f, flush=True)


        if ((count + 1) - NoAttack) !=0 and suc !=0:
            SR = suc / ((count + 1) - NoAttack)
            AverTime = time / suc
            AverChanged = Num_change / suc
            AverIter = Iter/suc
            AverQuery = Query/suc
            print("SuccessRate = %f, AverTime = %f, AverChanged = %f, AverIter= %f,AverQuery = %f,changed_words: %s" % (
                SR, AverTime, AverChanged,AverIter,AverQuery, str(changed_words)), file=log_f, flush=True)



if __name__ == '__main__':
    main()
