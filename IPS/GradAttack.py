import copy
import time
from utils import *
from model import *

import argparse

parser = argparse.ArgumentParser(description='malware')  # 创建parser对象
parser.add_argument('--budget', default=0, type=int, help='purturb budget')
parser.add_argument('--datatype', default='Normal', type=str, help='purturb budget')
parser.add_argument('--change', default=1, type=int, help='max change number in each iteration')
args = parser.parse_args()  # 解析参数，此处args是一个命名空间列表

class Attacker(object):
    def __init__(self, best_parameters_file, log_f):
        # def __init__(self, best_parameters_file, log_f, emb_weights):
        self.n_diagnosis_codes = 1104
        self.n_labels = n_lables

        self.model = RNN()
        # self.model = Net(dropout=False)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.load_state_dict(torch.load(best_parameters_file, map_location='cpu'))
        self.model.eval()

        self.log_f = log_f

        self.criterion = nn.CrossEntropyLoss()

    def input_handle(self, funccall, y):  # input:funccall, output:(seq_len,n_sample,m)[i][j][k]=k,
        funccall = [funccall]
        y = [y]
        t_diagnosis_codes, _ = pad_matrix(funccall, y, self.n_diagnosis_codes)
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
        label_set = {0, 1, 2}
        label_set.remove(y)
        list_label_set = list(label_set)
        g = logit[0][y]
        h1 = logit[0][list_label_set[0]]
        h2 = logit[0][list_label_set[1]]
        h = max(h1, h2)

        return pred, g, h, h1, h2

    def changed_set(self, eval_funccall, new_funccall):
        diff_set = set()
        for i in range(len(eval_funccall)):
            if eval_funccall[i] != new_funccall[i]:
                diff_set.add(i)
        return diff_set

    def eval_object(self, greedy_set, orig_label, greedy_set_best_temp_funccall, query_num):
        success_flag = 1
        label_set = {0, 1, 2}
        label_set.remove(orig_label)
        list_label_set = list(label_set)
        flip_set = set()
        flip_funccall = torch.tensor([])
        best_temp_funccall = greedy_set_best_temp_funccall

        self.model.train()
        query_num += 1
        change = False

        model_input, t_diagnosis_codes = self.input_handle(greedy_set_best_temp_funccall, orig_label)
        t_diagnosis_codes = torch.autograd.Variable(t_diagnosis_codes.data, requires_grad=True)
        logit = self.model(model_input, t_diagnosis_codes)
        loss = self.criterion(logit, torch.LongTensor([list_label_set[0]]).cuda())
        loss.backward()
        # subset_losses.append(loss)
        logit = logit.data.cpu().numpy()

        g = logit[0, orig_label]
        h = max(logit[0, list_label_set[0]], logit[0, list_label_set[1]])
        max_object = h - g
        grad_0 = t_diagnosis_codes.grad.cpu().data
        grad_0 = torch.abs(grad_0)

        batch_labels = [list_label_set[1]]
        logit = self.model(model_input, t_diagnosis_codes)
        loss = self.criterion(logit, torch.LongTensor(batch_labels).cuda())
        loss.backward()

        grad_1 = t_diagnosis_codes.grad.cpu().data
        grad_1 = torch.abs(grad_1)

        grad = torch.max(grad_0, grad_1)
        grad_norm = torch.norm(grad, 2, 2)
        grad_norm = grad_norm.reshape(-1, )
        list_greedy_set = list(greedy_set)
        for i in range(len(list_greedy_set)):
            grad_norm[i] = -1

        change_feature = torch.argsort(grad_norm, descending=True)[:min(change_num, budget-len(greedy_set))]
        candidate_list = [greedy_set_best_temp_funccall]
        for i in change_feature:
            candidate_temp = []
            for j in range(len(candidate_list)):
                for k in range(1104):
                    if candidate_list[j][i] != k:
                        temp_funccall = copy.deepcopy(candidate_list[j])
                        temp_funccall[i] = k
                        candidate_temp.append(temp_funccall)
            candidate_list = candidate_temp + candidate_list

        batch_size = 32
        query_num += len(candidate_list)
        n_batches = int(np.ceil(float(len(candidate_list)) / float(batch_size)))
        self.model.eval()
        for index in range(n_batches):  # n_batches

            batch_diagnosis_codes = candidate_list[batch_size * index: batch_size * (index + 1)]
            batch_labels = [orig_label] * len(batch_diagnosis_codes)
            t_diagnosis_codes, t_labels = pad_matrix(batch_diagnosis_codes, batch_labels, 1104)

            model_input = copy.copy(t_diagnosis_codes)
            for i in range(len(model_input)):
                for j in range(len(model_input[i])):
                    idx = 0
                    for k in range(len(model_input[i][j])):
                        model_input[i][j][k] = idx
                        idx += 1

            model_input = torch.FloatTensor(model_input).cuda()

            logit = self.model(model_input, torch.tensor(t_diagnosis_codes).cuda())
            logit = logit.data.cpu().numpy()
            subsets_g = logit[:, orig_label]
            subsets_h1 = logit[:, list_label_set[0]]
            subsets_h2 = logit[:, list_label_set[1]]
            subsets_h = np.max([subsets_h1, subsets_h2], axis=0)
            subsets_object = subsets_h - subsets_g
            max_temp_object = np.max(subsets_object)
            max_index = np.argmax(subsets_object)

            if max_temp_object >= max_object:
                max_object = max_temp_object
                change = True
                best_temp_funccall = copy.deepcopy(candidate_list[batch_size * index + max_index])

        if max_object >= 0:
            success_flag = 0
            flip_set = self.changed_set(greedy_set_best_temp_funccall, best_temp_funccall)
            flip_funccall = copy.deepcopy(best_temp_funccall)

        if not change:
            success_flag = -2
        changed_set = self.changed_set(greedy_set_best_temp_funccall, best_temp_funccall)
        greedy_set = greedy_set.union(changed_set)

        return max_object, best_temp_funccall, success_flag, greedy_set, flip_set, flip_funccall, query_num

    def attack(self, funccall, y):
        print()
        st = time.time()
        success_flag = 1

        orig_pred, orig_g, orig_h, orig_h1, orig_h2 = self.classify(funccall, y)

        greedy_set = set()
        greedy_set_best_temp_funccall = funccall
        risk_funccalls = set()
        flip_set = set()

        g_process = []
        mf_process = []
        greedy_set_process = []
        changed_set_process = []

        g_process.append(np.float(orig_g))
        mf_process.append(np.float(orig_h - orig_g))

        n_changed = 0
        iteration = 0
        robust_flag = 0
        query_num = 0

        current_object = orig_h - orig_g
        flip_funccall = funccall

        if current_object > 0:
            robust_flag = -1
            print("Original classification error")

            return g_process, mf_process, greedy_set_process, changed_set_process, \
                   query_num, robust_flag, greedy_set, greedy_set_best_temp_funccall, \
                   n_changed, flip_funccall, flip_set, risk_funccalls

        print(current_object)
        while success_flag == 1:
            iteration += 1

            worst_object, greedy_set_best_temp_funccall, success_flag, greedy_set, flip_set, flip_funccall, query_num \
                = self.eval_object(greedy_set, y, greedy_set_best_temp_funccall, query_num)

            print(iteration)
            print(worst_object)
            print(greedy_set)

            changed_set_process.append(self.changed_set(funccall, greedy_set_best_temp_funccall))
            pred, g, h, h1, h2 = self.classify(greedy_set_best_temp_funccall, y)
            g_process.append(np.float(g))
            mf_process.append(np.float(h - g))
            greedy_set_process.append(copy.deepcopy(greedy_set))

            if time.time() - st > 600 or success_flag == -2 or len(greedy_set) == budget:
                success_flag = -1
                robust_flag = 1

        n_changed = len(self.changed_set(funccall, greedy_set_best_temp_funccall))

        return g_process, mf_process, greedy_set_process, changed_set_process, \
               query_num, robust_flag, greedy_set, \
               greedy_set_best_temp_funccall, \
               n_changed, flip_funccall, flip_set, risk_funccalls


Model_Type = 'Gradattack'
Data_Type = args.datatype
budget = args.budget
change_num = args.change

print(Model_Type, Data_Type)
make_dir('./Logs/')
make_dir('./Logs/Gradattack/')
output_file = './Logs/Gradattack/'
if os.path.isdir(output_file):
    pass
else:
    os.mkdir(output_file)

data_file = 'dataset/mal_test_funccall.pickle'

test = pickle.load(open(data_file, 'rb'))

n_lables = 3

Model = {
    'Normal': './IPS_LSTM.par',
}

best_parameters_file = Model[Data_Type]

X = test

g_process_all = []
mf_process_all = []
greedy_set_process_all = []
changed_set_process_all = []

query_num_all = []
robust_flag_all = []

orignal_funccalls_all = []
orignal_labels_all = []

final_greedy_set_all = []
final_greedy_set_visit_idx_all = []
final_funccall_all = []
final_changed_num_all = []

flip_funccall_all = []
flip_set_all = []
flip_mf_all = []
flip_sample_original_label_all = []
flip_sample_index_all = []

# failue_set_all = []
risk_funccall_all = []
time_all = []

log_attack = open(
    output_file + 'grad_Attack.bak', 'w+')
attacker = Attacker(best_parameters_file, log_attack)
# attacker = Attacker(best_parameters_file, log_attack, emb_weights)
index = -1
label_file = 'dataset/mal_test_label.pickle'
label_file = pickle.load(open(label_file, 'rb'))
for i in range(len(X)):
    print(i)
    print("---------------------- %d --------------------" % i, file=log_attack, flush=True)

    sample = X[i]

    label = np.int(label_file[i])

    print('* Processing:%d/%d person' % (i, len(X)), file=log_attack, flush=True)

    print("* Original: " + str(sample), file=log_attack, flush=True)

    print("  Original label: %d" % label, file=log_attack, flush=True)

    st = time.time()
    best_g_process, best_mf_process, best_greedy_set_process, best_changed_set_process, query_num, robust_flag, \
    best_greedy_set, best_greedy_set_best_temp_funccall, \
    best_num_changed, best_flip_funccall, best_flip_set, best_risk_funccalls = attacker.attack(sample, label)
    print("Orig_Prob = " + str(best_g_process[0]), file=log_attack, flush=True)
    if robust_flag == -1:
        print('Original Classification Error', file=log_attack, flush=True)
    else:
        print("* Result: ", file=log_attack, flush=True)
    et = time.time()
    all_t = et - st

    if robust_flag == 1:
        print("This sample is robust.", file=log_attack, flush=True)

    if robust_flag != -1:
        index += 1
        print('g_process:', best_g_process, file=log_attack, flush=True)
        print('mf_process:', best_mf_process, file=log_attack, flush=True)
        print('greedy_set_process:', best_greedy_set_process, file=log_attack, flush=True)
        print("  Number of query for this: " + str(query_num), file=log_attack, flush=True)
        print('greedy_set: ', file=log_attack, flush=True)
        print(best_greedy_set, file=log_attack, flush=True)
        print('greedy_funccall:', file=log_attack, flush=True)
        print(best_greedy_set_best_temp_funccall, file=log_attack, flush=True)
        print('best_prob = ' + str(best_g_process[-1]), file=log_attack, flush=True)
        print('best_object = ' + str(best_mf_process[-1]), file=log_attack, flush=True)
        print("  Number of changed codes: %d" % best_num_changed, file=log_attack, flush=True)
        print("risk funccall:", file=log_attack, flush=True)
        print(best_risk_funccalls, file=log_attack, flush=True)
        print(" Time: " + str(all_t), file=log_attack, flush=True)
        if robust_flag == 0:
            print('flip_funccall:', file=log_attack, flush=True)
            print(best_flip_funccall, file=log_attack, flush=True)
            print('flip_set:', file=log_attack, flush=True)
            print(best_flip_set, file=log_attack, flush=True)
            print('flip_object = ', best_mf_process[-1], file=log_attack, flush=True)
            print(" The cardinality of S: " + str(len(best_greedy_set)), file=log_attack, flush=True)
        else:
            print(" The cardinality of S: " + str(len(best_greedy_set)) + ', but timeout', file=log_attack,
                  flush=True)

        time_all.append(all_t)
        g_process_all.append(copy.deepcopy(best_g_process))
        mf_process_all.append(copy.deepcopy(best_mf_process))
        greedy_set_process_all.append(copy.deepcopy(best_greedy_set_process))
        changed_set_process_all.append(copy.deepcopy(best_changed_set_process))

        query_num_all.append(query_num)
        robust_flag_all.append(robust_flag)

        orignal_funccalls_all.append(copy.deepcopy(X[i]))
        orignal_labels_all.append(label)

        final_greedy_set_all.append(copy.deepcopy(best_greedy_set))
        final_funccall_all.append(copy.deepcopy(best_greedy_set_best_temp_funccall))
        final_changed_num_all.append(best_num_changed)

        if robust_flag == 0:
            flip_funccall_all.append(copy.deepcopy(best_flip_funccall))
            flip_set_all.append(copy.deepcopy(best_flip_set))
            flip_mf_all.append(best_mf_process[-1])
            flip_sample_original_label_all.append(label)
            flip_sample_index_all.append(index)

        # failue_set_all.append(copy.deepcopy(best_failure_set))
        risk_funccall_all.append(copy.deepcopy(best_risk_funccalls))

pickle.dump(changed_set_process_all,
            open(output_file + 'gradattack_changed_set_process_%d_%d.pickle' % (budget, change_num), 'wb'))
pickle.dump(g_process_all,
            open(output_file + 'gradattack_g_process_%d_%d.pickle' % (budget, change_num), 'wb'))
pickle.dump(mf_process_all,
            open(output_file + 'gradattack_mf_process_%d_%d.pickle' % (budget, change_num), 'wb'))
pickle.dump(greedy_set_process_all,
            open(output_file + 'gradattack_greedy_set_process_%d_%d.pickle' % (budget, change_num), 'wb'))
pickle.dump(query_num_all,
            open(output_file + 'gradattack_querynum_%d_%d.pickle' % (budget, change_num), 'wb'))
pickle.dump(robust_flag_all,
            open(output_file + 'gradattack_robust_flag_%d_%d.pickle' % (budget, change_num), 'wb'))
pickle.dump(orignal_funccalls_all,
            open(output_file + 'gradattack_original_funccall_%d_%d.pickle' % (budget, change_num), 'wb'))
pickle.dump(orignal_labels_all,
            open(output_file + 'gradattack_original_label_%d_%d.pickle' % (budget, change_num), 'wb'))
pickle.dump(final_greedy_set_all,
            open(output_file + 'gradattack_greedy_set_%d_%d.pickle' % (budget, change_num), 'wb'))
pickle.dump(final_funccall_all,
            open(output_file + 'gradattack_modified_funccall_%d_%d.pickle' % (budget, change_num), 'wb'))
pickle.dump(final_changed_num_all,
            open(output_file + 'gradattack_changed_num_%d_%d.pickle' % (budget, change_num), 'wb'))
pickle.dump(flip_funccall_all,
            open(output_file + 'gradattack_flip_funccall_%d_%d.pickle' % (budget, change_num), 'wb'))
pickle.dump(flip_set_all,
            open(output_file + 'gradattack_flip_set_%d_%d.pickle' % (budget, change_num), 'wb'))
pickle.dump(flip_mf_all,
            open(output_file + 'gradattack_flip_mf_%d_%d.pickle' % (budget, change_num), 'wb'))
pickle.dump(flip_sample_original_label_all,
            open(output_file + 'gradattack_flip_sample_original_label_%d_%d.pickle' % (budget, change_num), 'wb'))
pickle.dump(flip_sample_index_all,
            open(output_file + 'gradattack_flip_sample_index_%d_%d.pickle' % (budget, change_num), 'wb'))
pickle.dump(risk_funccall_all,
            open(output_file + 'gradattack_risk_funccall_%d_%d.pickle' % (budget, change_num), 'wb'))
pickle.dump(time_all,
            open(output_file + 'gradattack_time_%d_%d.pickle' % (budget, change_num), 'wb'))

write_file('IPS', Model_Type, budget, 'grad_', 600)