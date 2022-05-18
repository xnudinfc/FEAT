from __future__ import print_function
import logging
import argparse
import torch.nn as nn
import time
from lm import NGramLangModel
from util import *
import spacy
import torch.nn.functional as F
import random
from tools import *
from copy import copy

# nlp = spacy.load('en', create_pipeline=wmd.WMD.create_spacy_pipeline)
nlp = spacy.load("en_core_web_lg")
NGRAM = 3
TAU = 0.8
N_NEIGHBOR = 50
N_REPLACE = 0
SECONDS = 1000
Budget = 6
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
    parser.add_argument('--model_path', action='store',
                        default='./model/yelpfull/train_lr_0.000100_b=16_lstm_{17}.bak', dest='model_path',
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
                if pred_probs.shape[1] == 2:
                    log_pred_prob, best_candidate_id = pred_probs[:, 1 - y].max(dim=0)
                else:
                    best_candidate_id = pred_probs[:, y].min(dim=0)[1]
                    log_pred_prob, pred = pred_probs[best_candidate_id, :].max(dim=0)

                new_words = candidates[best_candidate_id.data]
                pred_prob = exp(log_pred_prob.data)
            elif self.opt.model == 'CNN':
                candidate_var = self.text_to_var_CNN(candidates, self.src_vocab)
                pred_probs, _ = self.model(candidate_var)
                log_pred_prob, best_candidate_id = pred_probs[:, 1 - y].max(dim=0)
                new_words = candidates[best_candidate_id.data[0]]
                pred_prob = log_pred_prob.data[0]
        else:
            print('empty candidates!')
        return new_words, pred_prob, j, pred

    def word_select(self, words, poses, list_neighbors, y):
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
                if pred_probs.shape[1] == 2:
                    log_pred_prob, best_candidate_id = pred_probs[:, 1 - y].max(dim=0)
                else:
                    best_candidate_id = pred_probs[:, y].min(dim=0)[1]
                    log_pred_prob, pred = pred_probs[best_candidate_id, :].max(dim=0)

                new_words = candidates[best_candidate_id.data]
                pred_prob = exp(log_pred_prob.data)

        else:
            print('empty candidates!')
        return new_words, pred_prob, j, pred

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


    def one_code(self,words,list_closest_neighbors,y,set_ind):
        candidates = [words]
        index_candidate = np.zeros(len(words),dtype = int)
        pred_candidate = []

        for index, cand_word_list in enumerate(list_closest_neighbors):
            for i in set_ind:
                if index == i:
                    for j, cand_word in enumerate(cand_word_list):
                        # print(index,cand_word_list,j,cand_word)

                        corrupted = copy(words)
                        corrupted[index] = cand_word
                        candidates.append(corrupted)
                # print(len(candidates))
                    # choose the neighbors best
                    if candidates:
                        if self.opt.model == 'LSTM':
                            candidate_var = text_to_var(candidates, self.src_vocab)
                            pred_probs = self.model(candidate_var)
                            # print('pred_probs',pred_probs.shape,pred_probs)
                            pred_probs = torch.exp(pred_probs)
                            # print('pred_probs', pred_probs)
                            if pred_probs.shape[1] == 2:
                                pred_prob, best_candidate_id = pred_probs[:, 1 - y].max(dim=0)
                            else:
                                best_candidate_id = pred_probs[:, y].min(dim=0)[1]
                                pred_probs[:, y] = 0
                                pred_prob, pred = pred_probs[best_candidate_id, :].max(dim=0)

                            # print('pred_prob', pred_prob, pred)
                            # new_words = candidates[best_candidate_id.data]

                            index_candidate[index] = best_candidate_id
                            pred_candidate.append(pred_prob.data.cpu().item())
                            # print(best_candidate_id, pred_prob)

                predmax = np.max(pred_candidate)  # this is the sentence level optimal prediction
                word_best_id = np.argmax(pred_candidate)




        # recover the best word
        index_best_candit_word = index_candidate[word_best_id]
        rel = list_closest_neighbors[word_best_id][index_best_candit_word]
        new_words = copy(words)
        new_words[index_best_candit_word] = rel

        return predmax,word_best_id,index_best_candit_word,rel,new_words


    def New_Word(self,words,arm_chain,list_closest_neighbors):
        new_word = copy(words)
        if len(arm_chain) == 0:
            pass
        else:
            for pos in arm_chain:
                if pos[1] != 0:
                    new_word[pos[0]] = list_closest_neighbors[pos[0]][pos[1] - 1]
        return new_word

    def word_paraphrase_nocombine(self, new_words,orig_pred_new, poses, list_closest_neighbors, y):
        index_candidate = np.zeros(len(poses), dtype=int)
        pred_candidate = np.zeros(len(poses))
        for i,candi_sample_index in enumerate(poses):
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


    def WordNeigbor_select(self,new_words, orig_pred_new, K_set,list_closest_neighbors,y,arm_chain):
        index_candidate, pred_candidate= self.word_paraphrase_nocombine(new_words, orig_pred_new, K_set,
                                                                                     list_closest_neighbors,y)

        topk_feature_index = np.argsort(pred_candidate)[-1]

        pred_max = pred_candidate[topk_feature_index]
        Feat_max = K_set[topk_feature_index]
        cand_max = index_candidate[topk_feature_index]



        return (Feat_max, cand_max),pred_max


    def attack(self, count, doc, y, NoAttack,log_f):
        # ---------------------------------word paraphrasing----------------------------------------------#
        words = doc.split()
        words_before = copy(words)
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


        changed_pos = set()
        iteration = 0
        recompute = True
        n_change = 0
        if self.opt.model == 'CNN':
            lword = min(len(words), self.model.sentence_len)
        else:
            lword = len(words)

        allCode_ori = list(range(len(words)))
        allCode = []
        for code_ind in allCode_ori:
            if len(list_closest_neighbors[code_ind]) != 0:
                allCode.append(code_ind)

        allCode = list(range(len(words)))
        Set_candi_sample = []
        iteration = 0
        words_before = copy(words)
        changed_words = []

        Set_c = [()]
        Set_best = []
        Set_residual = DiffElem(allCode, Set_best)
        pred_best = best_score
        robust_flag =1
        ## step2: this is attack main part.
        st = time.time()
        while robust_flag == 1 and n_change <= Budget and time.time() - st <= SECONDS:  # no attack success condition only 0.2 original feature
            iteration += 1

            Pred_select = []
            Set_select = []
            arm_chain = []
            for set_ in Set_c:  # all the combination set
                if len(set_)==0:
                    new_words = copy(words)
                else:
                    new_words = copy(words)
                    new_words = self.New_Word(new_words,set_,list_closest_neighbors)

                new_words_var = text_to_var([new_words], self.src_vocab)
                orig_prob_new, orig_pred_new = classify(new_words_var, self.model)

                (Feat_max, cand_max), pred_max = self.WordNeigbor_select(new_words, orig_pred_new,Set_residual,
                                                                            list_closest_neighbors, y, set_ )
                # if (feature_n > len(Set_residual) - 1) and (
                #         (Feat_max, cand_max) in arm_chain or (Feat_max, cand_max) == (Feat_max, 0)):
                #     break

                if pred_max >= best_score:
                    Set_select.append(set_)
                    arm_chain.append((Feat_max, cand_max))
                    Pred_select.append(pred_max)
                    best_score = pred_max

            if len(arm_chain) != 0:
                pred_best = Pred_select[-1]
                new_words = self.New_Word(new_words, [arm_chain[-1]], list_closest_neighbors)
                Set_best.append(arm_chain[-1])
            else:
                pred_best = best_score
                new_words = new_words
                Set_best.append((Feat_max, cand_max))

            n_change = sum([0 if words_before[i] == new_words[i] else 1 for i in range(len(words))])
            Set_residual = DiffElem(allCode, [Set_best[-1][0]])
            Set_c = list(powerset(Set_best))

            if pred_best >= TAU:
                robust_flag = 0

        Time = time.time() - st
        for i in range(len(words)):
            if words_before[i] != new_words[i]:
                changed_words.append([words_before[i],new_words[i]])
        query_num = len(allCode) * (iteration +1)/2
        return new_words, pred_best, n_change, changed_words, Time,  iteration, NoAttack,query_num

def main():
    opt = parse_args()
    log_f = open(
        './Logs/Robust_Mf/FSGS_L_30_200_Budget=%s_N=%s_TAU=%s_N_REPLACE=%d_SECONDS=%s.bak' % (str(Budget),str(N_NEIGHBOR), str(TAU), N_REPLACE,str(SECONDS)),
        'w+')

    TITLE = '=== ' + 'FSGS' + ' target prob = ' + str(TAU) + ' changeRange = ' + str(N_REPLACE) + ' ==='

    print(TITLE)
    print(TITLE, file=log_f, flush=True)
    X_train, y_train = read_data_multi_class(opt.train_path)
    # X, y = read_data_multi_class(opt.test_path)
    [X, y] = pickle.load(open('./data/YelpFull/test_selected_long_balanced30_200.pkl', 'rb'))
    attacker = Attacker(X_train, opt)
    del X_train
    del y_train
    suc = 0
    time = 0
    Num_change = 0
    Iter = 0
    Query = 0
    NoAttack = 0
    suffix = 'wordonly-' + str(opt.word_delta)

    for count, doc in enumerate(X):
        # logging.info("Processing %d/%d documents", count + 1, len(X))
        print("====Sample %d/%d ======", count + 1, len(X), file=log_f, flush=True)
        changed_doc, flag, num_changed, changed_words, Time, iteration, NoAttack,query_num = attacker.attack(count, doc,
                                                                                                         y[count],
                                                                                                         NoAttack,log_f)
        print(" The cost Time", Time, file=log_f, flush=True)
        print(" The number changed", num_changed, file=log_f, flush=True)
        print(" The changed words", changed_words, file=log_f, flush=True)
        print(" NoAttack Number", NoAttack, file=log_f, flush=True)
        try:
            v = float(flag)
            if v > TAU:
                suc += 1
                time = time + Time
                changed_y = y[count]
                Num_change = Num_change + num_changed
                Iter = Iter + iteration
                Query = Query + query_num
            else:
                continue
        except:
            changed_y = 1
        if suc != 0 and ((count + 1) - NoAttack) != 0:
            SR = suc / ((count + 1) - NoAttack)
            AverTime = time / suc
            AverChanged = Num_change / suc
            AverIter = Iter/suc
            AverQuery = Query/suc
            print("SuccessRate = %f, AverTime = %f, AverChanged = %f,AverIter = %f,AverQuery = %fchanged_words: %s" % (
                SR, AverTime, AverChanged,AverIter,AverQuery, str(changed_words)), file=log_f, flush=True)

if __name__ == '__main__':
    main()
