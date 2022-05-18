import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data
from sklearn.metrics import accuracy_score

from lstm import LSTMClassifier
device = -1
if torch.cuda.is_available():
    device = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', action='store', default='./data/YelpFull/train.tsv',type=str,dest='train_path',
                        help='Path to train data')
    parser.add_argument('--test_path', action='store', default='./data/YelpFull/test.tsv',type=str,dest='test_path',
                        help='Path to test.txt data')
    parser.add_argument('--log-every', type=int, default=10000, help='Steps for each logging.')

    parser.add_argument('--batch-size', action='store', default=16, type=int,
                        help='Mini batch size.')
    parser.add_argument('--lr', action='store', default= 0.0001, type=float,
                        help='learning rate.')

    return parser.parse_args()

def evaluate(model, batch):
    inputs=batch.text #F.pad(batch.text.transpose(0,1), (0,sentence_len-len(batch.text)))
    preds =model(inputs.cuda())
    #print(preds.data.cpu().numpy(), batch.label.data.cpu().numpy())
    eval_acc=sum([1 if preds.data.cpu().numpy()[i][j]>-0.693147181 else 0 for i,j in enumerate(batch.label.data.cpu().numpy()[0])])
    return eval_acc

import csv
def re_order(in_file, out_file):
    with open(in_file, 'r', newline='') as in_file_handle:
        reader = csv.reader(in_file_handle)
        columns = [1, 0]
        content = []
        for row in reader:
            content.append([row[1],row[0]])
        with open(out_file, 'w', newline='') as out_file_handle:
            writer = csv.writer(out_file_handle,delimiter='\t')
            for i in content:
                writer.writerow(i)
            # writer.writerow(content)


def main():

    opt = parse_args()
    log_f = open('./Logs/TEXTFoolerTESTForori_train_lr_%f_b=%d.bak'%(opt.lr,opt.batch_size), 'w+')
    TITLE = '===== ' + 'train_learning Rate'+ str(opt.lr)+' _ Batch Size' +str(opt.batch_size) +' ====='
    print(TITLE)
    print(TITLE,file=log_f, flush=True)
    src_field = data.Field()
    label_field = data.Field(pad_token=None, unk_token=None)
    import pandas as pd
    #'this is for creating proper data source '
    # re_order('.data/yelp_review_full_csv/train.csv', './data/YelpFull/train.tsv')

    train = data.TabularDataset(
        path=opt.train_path, format='tsv',
        fields=[('text', src_field), ('label', label_field)]
    )

    test = data.TabularDataset(
        path=opt.test_path, format='tsv',
        fields=[('text', src_field), ('label', label_field)]
    )
    src_field.build_vocab(train, max_size=100000, min_freq=2, vectors="glove.6B.300d")
    label_field.build_vocab(train)

    print("Training size: {0}, Testing size: {1}".format(len(train), len(test)),file=log_f, flush=True)

    classifier = LSTMClassifier(300, 512, len(label_field.vocab), src_field.vocab.vectors)

    train_iter = data.BucketIterator(
        dataset=train,
        batch_size=opt.batch_size,
        device=device,
        repeat=False
    )
    test_iter = data.BucketIterator(
        dataset=test,
        batch_size=5,
        device=device,
        repeat=False
    )

    if torch.cuda.is_available():
        classifier.cuda()
        for param in classifier.parameters():
            param.data.uniform_(-0.08, 0.08)
        # classifier = torch.load('./model/yelpfull/lstm_9')
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam(classifier.parameters(),lr=opt.lr)


    step = 0
    for epoch in range(0,100):
        test_acc=0
        for batch in test_iter:
            test_acc+=evaluate(classifier, batch)
        print('Test accuracy: {0}'.format(test_acc/len(test)),file=log_f, flush=True)
        running_loss = 0.0
        train_acc = 0
        for batch in train_iter:
            optimizer.zero_grad()
            pred = classifier(batch.text.cuda())
            loss = criterion(pred, (batch.label.view(-1)).cuda())
            running_loss += loss.data
            loss.backward()
            optimizer.step()
            step += 1
            if step % opt.log_every == 0:
                print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, step + 1, running_loss / opt.log_every),file=log_f, flush=True)
                running_loss = 0.0
            train_acc+=evaluate(classifier, batch)

        print('Train accuracy: {0}'.format(train_acc/len(train)),file=log_f, flush=True)
        torch.save(classifier, os.path.join('./model/yelpfull/TESTori_train_lr_%f_b=%d_lstm_{%d}.bak'%(opt.lr,opt.batch_size,epoch+1)))

if __name__ == '__main__':
    main()


