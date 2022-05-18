import argparse
import random
import torch
import copy
import pickle
import rootpath
rootpath.append()
import os
import numpy as np
import rnn_tools as tools
from torch.autograd import Variable
import time
import torch.nn as nn

import rnn_model as model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

parser = argparse.ArgumentParser(description='EHR with LSTM in PyTorch ')    #创建parser对象
parser.add_argument('--model_name', default='LSTM', type=str, help='model_name')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--log_eps', default=1e-8, type=float, help='log_eps for optimizer')
parser.add_argument('--L2_reg', default=0.001, type=float, help='L2_reg for optimizer')
parser.add_argument('--n_epoch', default=100, type=int, help='number of epochs tp train for')
parser.add_argument('--batch_size', default=1, type=int, help='batch_size')
parser.add_argument('--dropout_rate', default=0.5, type=float, help='dropout_rate')

parser.add_argument('--threshold', type=int, default=0, help='threshold')

parser.add_argument('--trianing_file', default='./DataSource/hf_dataset_training.pickle',type=str, help='trianing_file')
parser.add_argument('--validation_file', default='./DataSource/hf_dataset_validation.pickle',type=str, help='validation_file')
parser.add_argument('--testing_file', default='./DataSource/hf_dataset_testing.pickle',type=str, help='testing_file')

parser.add_argument('--n_labels', default=2, type=int, help='binary classification')
parser.add_argument('--visit_size', default=70, type=int, help='visit_size')
parser.add_argument('--hidden_size', default=70, type=int, help='hidden_size')
parser.add_argument('--n_diagnosis_codes', default=4130, type=int, help='n_diagnosis_codes')
parser.add_argument('--n_claims', default=504, type=int, help='n_claims')
parser.add_argument('--seed', type=int, default=666, help='random seed (default: 100)')


args=parser.parse_args()#解析参数，此处args是一个命名空间列表
# torch.manual_seed(args.seed)
# random.seed(args.seed)
# torch.cuda.manual_seed_all(args.seed)
print(args)
def simplex_projection(s):
    """Projection onto the unit simplex."""
    if np.sum(s) <=1 and np.alltrue(s >= 0):
        return s
    # Code taken from https://gist.github.com/daien/1272551
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(s)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, len(u)+1) > (cssv - 1))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - 1) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    return np.maximum(s-theta, 0)

def nuclear_projection(A):
    """Projection onto nuclear norm ball."""
    U, s, V = np.linalg.svd(A, full_matrices=False)
    s = simplex_projection(s)
    return U.dot(np.diag(s).dot(V))

def train_model(emb_weights = 1,
                training_file='training_file',
                validation_file='validation_file',
                testing_file='testing_file',
                n_diagnosis_codes=10000,
                n_labels=2,
                output_file='output_file',
                batch_size=100,
                dropout_rate=0.9,
                lr=0.001,
                n_epoch=1000,
                log_eps=1e-8,
                n_claims=300,
                visit_size=512,
                hidden_size=256,
                model_name=''):
    options = locals().copy()
    log_f = open(
        './Logs/%s_lr=%s.bak' % (
            args.model_name, str(args.lr)), 'w+')
    print('building the model ...', file=log_f, flush=True)

    rnn = model.LSTM(options, emb_weights)
    rnn = rnn.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    # rnn.load_state_dict(torch.load('Outputs/rnnori/1e-06.99'))

    print('constructing the optimizer ...', file=log_f, flush=True)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=options['lr'])
    print('done!', file=log_f, flush=True)

    print('loading data ...', file=log_f, flush=True)
    train, validate, test = tools.load_data(training_file, validation_file, testing_file)
    n_batches = int(np.ceil(float(len(train[0])) / float(batch_size)))
    # n_batches = 3
    print('training start', file=log_f, flush=True)
    best_train_cost = 0.0
    best_validate_cost = 100000000.0
    best_test_cost = 0.0
    epoch_duaration = 0.0
    best_epoch = 0.0
    best_parameters_file = ''

    maxmax = 0

    for epoch in range(0,n_epoch):
        iteration = 0
        cost_vector = []
        start_time = time.time()
        samples = random.sample(range(n_batches), n_batches)
        counter = 0

        rnn.train()
        for index in samples:
            batch_diagnosis_codes = train[0][batch_size * index: batch_size * (index + 1)]
            batch_labels = train[1][batch_size * index: batch_size * (index + 1)]

            t_diagnosis_codes, t_labels, t_mask = tools.pad_matrix(batch_diagnosis_codes, batch_labels, options)

            model_input = copy.copy(t_diagnosis_codes)
            for i in range(len(model_input)):
                for j in range(len(model_input[i])):
                    idx = 0
                    for k in range(len(model_input[i][j])):
                        model_input[i][j][k] = idx
                        idx += 1

            model_input = Variable(torch.LongTensor(model_input).cuda())
            t_labels = Variable(torch.LongTensor(t_labels).cuda())
            t_diagnosis_codes = Variable(torch.tensor(t_diagnosis_codes).cuda())

            optimizer.zero_grad()

            logit = rnn(model_input, t_diagnosis_codes)

            loss = criterion(logit, t_labels)
            loss.backward()
            optimizer.step()

            if args.model_type == 'sub':
                for p in rnn.lstm.parameters():
                        p.data = abs(p.data)

            if args.model_type== 'nuclear_norm':
                for p in rnn.lstm.parameters():
                    # print(p.shape,p)
                    if len(p.shape) == 2:
                        p.data = torch.FloatTensor(nuclear_projection(p.cpu().detach().numpy())).cuda()
                    if len(p.shape) == 4:
                        grad_u = np.empty([1, 1, 20, 20], dtype='float32')
                        for i in range(len(p.shape)):
                            a = torch.reshape(p[i, 0, :, :], [20, 20]).cpu().detach().numpy()
                            each = nuclear_projection(a)
                            each = np.reshape(each, [1, 1, 20, 20])
                            grad_u = np.concatenate((grad_u, each), axis=0)

                        p.data = torch.FloatTensor(grad_u[1:5, :, :, :]).cuda()

            cost_vector.append(loss.cpu().data.numpy())

            if (iteration % 1000 == 0):
                print('epoch:%d, iteration:%d/%d, cost:%f' % (epoch, iteration, n_batches, loss.cpu().data.numpy()),
                      file=log_f, flush=True)
            iteration += 1

        duration = time.time() - start_time
        torch.save(rnn.state_dict(), output_file + '.' + str(epoch))
        print('epoch:%d, mean_cost:%f, duration:%f' % (epoch, np.mean(cost_vector), duration), file=log_f, flush=True)

        train_cost = np.mean(cost_vector)
        validate_cost = tools.calculate_cost(rnn, validate, options)
        epoch_duaration += duration

        if validate_cost < best_validate_cost:
            best_validate_cost = validate_cost
            best_train_cost = train_cost
            best_epoch = epoch

        buf = 'Best Epoch:%d, Train_Cost:%f, Valid_Cost:%f' % (best_epoch, best_train_cost, best_validate_cost)
        print(buf, file=log_f, flush=True)

        # testing
        print('-----------test--------------', file=log_f, flush=True)
        best_parameters_file = output_file + '.' + str(best_epoch)

        # best_parameters_file ='/home/baoh/PycharmProjects/CertificatedRobust/EHR/classifier/Outputs/Nonsubmodular_lstm.42'
        rnn.load_state_dict(torch.load(best_parameters_file))
        rnn.eval()

        n_batches_test = int(np.ceil(float(len(test[0])) / float(batch_size)))
        y_true = np.array([])
        y_pred = np.array([])

        for index in range(n_batches_test):
            batch_diagnosis_codes = test[0][batch_size * index: batch_size * (index + 1)]
            batch_labels = test[1][batch_size * index: batch_size * (index + 1)]
            t_diagnosis_codes, t_labels, t_mask = tools.pad_matrix(batch_diagnosis_codes, batch_labels, options)

            model_input = copy.copy(t_diagnosis_codes)
            for i in range(len(model_input)):
                for j in range(len(model_input[i])):
                    idx = 0
                    for k in range(len(model_input[i][j])):
                        model_input[i][j][k] = idx
                        idx += 1


            model_input = Variable(torch.LongTensor(model_input).cuda())
            t_diagnosis_codes = Variable(torch.tensor(t_diagnosis_codes).cuda())

            logit = rnn(model_input, t_diagnosis_codes)


            prediction = torch.max(logit, 1)[1].view((len(t_labels),)).data.cpu().numpy()


            y_true = np.concatenate((y_true, t_labels))
            y_pred = np.concatenate((y_pred, prediction))

        accuary = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)
        print('accuary:, precision:, recall:, f1:,auc:', (accuary, precision, recall, f1, roc_auc), file=log_f,
              flush=True)

    return accuary, precision, recall, f1, roc_auc


def main():
    emb_weights_word = torch.load("./DataSource/PretrainedEmbedding.4")['word_embeddings.weight']
    # emb_char_word_Mix = pickle.load(open("./DataSource/char_word_MixEmbed.pickle", 'rb'))
    output_file_path = './Outputs/'+ args.model_name+"/" + str(
        args.lr)

    # emb_weights = emb_char_word_Mix
    emb_weights = emb_weights_word

    accuary, precision, recall, f1, roc_auc = train_model(emb_weights, args.trianing_file, args.validation_file,
                                                          args.testing_file, args.n_diagnosis_codes, args.n_labels,
                                                          output_file_path, args.batch_size, args.dropout_rate,
                                                          args.lr, args.n_epoch, args.log_eps, args.n_claims, args.visit_size, args.hidden_size, args.model_name)
if __name__ == '__main__':
    main()
    print("Well Done")