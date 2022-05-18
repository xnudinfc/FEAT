from tools import *
import torch.optim as optim
from model import *
import random
import time
import argparse

parser = argparse.ArgumentParser(description='PEDec with CNN')    #创建parser对象
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--batch_size', default=4, type=int, help='batch_size')
parser.add_argument('--dropout_rate', default=0.5, type=float, help='dropout_rate')
parser.add_argument('--n_epoch', default=100, type=int, help='number of epochs tp train for')
args=parser.parse_args()#解析参数，此处args是一个命名空间列表
print(args)

def pre_data(discreteData,label):

    discreteData_weight = np.where(discreteData == 0, 0.01, 0.99)
    data_input_norm = discreteData_weight
    # divide input data to train and test data
    pos_idx = np.where(label == 1)[0]
    neg_idx = np.where(label == 0)[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    train_idx = pos_idx[:int(float(len(pos_idx)) * 0.8)].tolist() + neg_idx[:int(float(len(neg_idx)) * 0.8)].tolist()
    test_idx = pos_idx[int(float(len(pos_idx)) * 0.8):].tolist() + neg_idx[int(float(len(neg_idx)) * 0.8):].tolist()
    train_data = np.array(data_input_norm)[np.array(train_idx)]
    train_label = label[train_idx]
    test_data = np.array(data_input_norm)[np.array(test_idx)]
    test_label = label[test_idx]

    return train_data,train_label,test_data,test_label,


def test_model(model, test_data,test_label, batch_size):
    n_batches = int(np.ceil(float(len(test_label)) / float(batch_size)))
    cost_sum = 0.0
    correct = 0
    total = 0
    for index in range(n_batches):
        batch_test_data = test_data[batch_size * index: batch_size * (index + 1)]
        batch_test_label = test_label[batch_size * index: batch_size * (index + 1)]

        batch_num = batch_test_data.shape[0]
        weight = torch.unsqueeze(torch.tensor(batch_test_data), dim=1).cuda()
        logit = model(weight).cuda()

        loss = F.cross_entropy(logit, torch.LongTensor(batch_test_label).cuda())
        cost_sum += loss.cpu().data.numpy()

        pred = torch.argmax(logit, 1)
        correct += (pred == torch.LongTensor(batch_test_label).cuda()).sum().item()
        total += batch_num

    Valid_Accu = 100 * correct / total
    return cost_sum / n_batches,Valid_Accu

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

def train_model(n_epoch,batch_size):
    print('build the CNN model', file=log_f, flush=True)
    net = Net_0D(num_uniqFeature).cuda()
    print('constructing the optimizer ...', file=log_f, flush=True)
    CEloss = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    print('loading data', file=log_f, flush=True)
    train_data,train_label,test_data,test_label = pre_data(data,label)

    n_batches = int(np.ceil(float(len(train_label)) / float(batch_size)))

    print('training start', file=log_f, flush=True)
    best_train_cost = 0.0
    best_validate_cost = 100000000.0
    best_test_cost = 0.0
    epoch_duaration = 0.0
    best_epoch = 0.0
    best_parameters_file = ''
    net.train()
    for epoch in range(0,n_epoch):
        # print('------------------------------------Epoch---------------------------------',epoch)
        iteration = 0
        cost_vector = []
        start_time = time.time()
        samples = random.sample(range(n_batches), n_batches)
        total = 0
        correct = 0
        i = 0
        for index in samples:
            # print('------------------------------------index---------------------------------',i)
            i = i+1
            batch_train_data = train_data[batch_size * index : batch_size * (index + 1)]
            batch_train_label = train_label[batch_size * index : batch_size * (index + 1)]

            batch_num = batch_train_data.shape[0]
            weight = torch.unsqueeze(torch.tensor(batch_train_data), dim=1).cuda()
            # if need grad
            weight.requires_grad_()

            logit = net(weight).cuda()
            loss = CEloss(logit, torch.LongTensor(batch_train_label).cuda())
            loss.backward()
            # print(loss,torch.min(weight.grad),torch.max(weight.grad))
            optimizer.step()

            if model_name == 'sub':
                for p in net.parameters():
                    p.data = abs(p.data)
            if model_name == 'nuclear':
                for p in net.parameters():
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
            pred = torch.argmax(logit, 1)
            correct += (pred == torch.LongTensor(batch_train_label).cuda()).sum().item()
            total += batch_num

            if (iteration % 200 == 0):
                print('epoch:%d, iteration:%d/%d, cost:%f,duration:%f' % (epoch, iteration, n_batches, loss.cpu().data.numpy(),time.time() - start_time), file=log_f, flush=True)
                # print(pred)
            iteration += 1

        Train_Accu = 100 * correct / total
        duration = time.time() - start_time
        torch.save(net.state_dict(), output_file+str(lr) +'.' + str(epoch))
        print('epoch:%d, mean_cost:%f,accuracy:%f, duration:%f' % (epoch, np.mean(cost_vector), Train_Accu,duration), file=log_f, flush=True)

        train_cost = np.mean(cost_vector)
        net.eval()
        validate_cost,Valid_Accu = test_model(net, test_data,test_label, batch_size)
        epoch_duaration += duration

        if validate_cost < best_validate_cost:
            best_validate_cost = validate_cost
            best_train_cost = train_cost
            best_epoch = epoch
            best_Train_Accu = Train_Accu
            best_Valid_Accu = Valid_Accu

        buf = 'Best Epoch:%d, Train_Cost:%f, Valid_Cost:%f, Train_Accu:%f, Valid_Accu:%f' % (best_epoch, best_train_cost, best_validate_cost,best_Train_Accu,best_Valid_Accu)
        print(buf, file=log_f, flush=True)
if __name__ == '__main__':
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    discreteData_path = 'dataset/5000_data.pickle'
    discreteData = load_data(discreteData_path)
    data = discreteData[0]
    label = discreteData[1]
    model_name = args.model
    lr = args.lr
    batch_size = args.batch_size
    n_epoch = args.n_epoch
    output_file = './Output/' + model_name + '/'
    num_uniqFeature = len(discreteData[0][0])

    log_f = open('./Logs/%s_k=%f.bak' % (model_name, lr), 'w+')



    train_model(n_epoch, batch_size)
    train_model(n_epoch,batch_size)
    print("Well Done")

print('Finished Training', file=log_f, flush=True)

print('end', file=log_f, flush=True)