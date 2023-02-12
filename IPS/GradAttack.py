import copy
import time
from utils import *
from model import *

import argparse

parser = argparse.ArgumentParser(description='malware')
parser.add_argument('--budget', default=5, type=int, help='purturb budget')
parser.add_argument('--datatype', default='Normal', type=str, help='purturb budget')
parser.add_argument('--change', default=1, type=int, help='max change number in each iteration')
args = parser.parse_args()

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

    def input_handle(self, funccall):  # input:funccall, output:(seq_len,n_sample,m)
        # put the funccall and label into a list
        funccall = torch.LongTensor([funccall])
        # change the list to one hot vectors
        t_diagnosis_codes = input_process(funccall, 'IPS')
        return torch.tensor(t_diagnosis_codes).cuda()

    def classify(self, funccall, y):
        weight_of_embed_codes = self.input_handle(funccall)
        logit = self.model(weight_of_embed_codes)
        logit = logit.cpu()
        # get the prediction
        pred = torch.max(logit, 1)[1].view((1,)).data.numpy()
        logit = logit.data.cpu().numpy()
        # get the false labels
        label_set = set(range(self.n_labels))
        label_set.remove(y)
        list_label_set = list(label_set)
        g = logit[0][y]
        # find the largest prediction in the false labels
        h = max([logit[0][false_class] for false_class in list_label_set])
        return pred, g, h

    def changed_set(self, eval_funccall, new_funccall):
        diff_set = set(np.where(eval_funccall != new_funccall)[0])
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

        t_diagnosis_codes = self.input_handle(greedy_set_best_temp_funccall)
        t_diagnosis_codes = torch.autograd.Variable(t_diagnosis_codes.data, requires_grad=True)
        logit = self.model(t_diagnosis_codes)
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
        logit = self.model(t_diagnosis_codes)
        loss = self.criterion(logit, torch.LongTensor(batch_labels).cuda())
        loss.backward()

        grad_1 = t_diagnosis_codes.grad.cpu().data
        grad_1 = torch.abs(grad_1)

        grad = torch.max(grad_0, grad_1)
        grad_norm = torch.norm(grad, 2, 2)
        grad_norm = grad_norm.reshape(-1, )
        for i in greedy_set:
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

        batch_size = 64
        query_num += len(candidate_list)
        n_batches = int(np.ceil(float(len(candidate_list)) / float(batch_size)))
        self.model.eval()
        for index in range(n_batches):  # n_batches

            batch_diagnosis_codes = torch.LongTensor(candidate_list[batch_size * index: batch_size * (index + 1)])
            t_diagnosis_codes = input_process(batch_diagnosis_codes, 'IPS')
            logit = self.model(torch.tensor(t_diagnosis_codes).cuda())
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

        orig_pred, orig_g, orig_h = self.classify(funccall, y)

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
                   query_num, robust_flag, greedy_set, \
                   greedy_set_best_temp_funccall, \
                   n_changed, flip_funccall, flip_set, iteration

        print(current_object)
        while success_flag == 1:
            iteration += 1

            worst_object, greedy_set_best_temp_funccall, success_flag, greedy_set, flip_set, flip_funccall, query_num \
                = self.eval_object(greedy_set, y, greedy_set_best_temp_funccall, query_num)

            print(iteration)
            print(worst_object)
            print(greedy_set)

            changed_set_process.append(self.changed_set(funccall, greedy_set_best_temp_funccall))
            pred, g, h = self.classify(greedy_set_best_temp_funccall, y)
            g_process.append(np.float(g))
            mf_process.append(np.float(h - g))
            greedy_set_process.append(copy.deepcopy(greedy_set))

            if time.time() - st > 10 or success_flag == -2 or len(greedy_set) == budget:
                success_flag = -1
                robust_flag = 1

        n_changed = len(self.changed_set(funccall, greedy_set_best_temp_funccall))

        return g_process, mf_process, greedy_set_process, changed_set_process, \
               query_num, robust_flag, greedy_set, \
               greedy_set_best_temp_funccall, \
               n_changed, flip_funccall, flip_set, iteration


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

X, y = load_data()

n_lables = 3
best_parameters_file = 'IPS_LSTM.par'

g_process_all = []
mf_process_all = []
greedy_set_process_all = []
changed_set_process_all = []

query_num_all = []
robust_flag_all = []

orignal_funccalls_all = []
orignal_labels_all = []

final_greedy_set_all = []
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
iteration_all = []

log_attack = open(
    output_file + 'grad_Attack.bak', 'w+')
attacker = Attacker(best_parameters_file, log_attack)
# attacker = Attacker(best_parameters_file, log_attack, emb_weights)
index = -1
robust = 0
for i in range(len(X)):
    print(i)
    print("---------------------- %d --------------------" % i, file=log_attack, flush=True)

    sample = X[i]

    label = np.int(y[i])

    print('* Processing:%d/%d person' % (i, len(X)), file=log_attack, flush=True)

    print("* Original: " + str(sample), file=log_attack, flush=True)

    print("  Original label: %d" % label, file=log_attack, flush=True)

    st = time.time()
    g_process, mf_process, greedy_set_process, changed_set_process, query_num, robust_flag, \
    greedy_set, greedy_set_best_temp_funccall, \
    num_changed, flip_funccall, flip_set, iteration = attacker.attack(sample, label)
    print("Orig_Prob = " + str(g_process[0]), file=log_attack, flush=True)
    if robust_flag == -1:
        print('Original Classification Error', file=log_attack, flush=True)
    else:
        print("* Result: ", file=log_attack, flush=True)
    et = time.time()
    all_t = et - st

    if robust_flag == 1:
        print("This sample is robust.", file=log_attack, flush=True)
        robust += 1

    if robust_flag != -1:
        print('g_process:', g_process, file=log_attack, flush=True)
        print('mf_process:', mf_process, file=log_attack, flush=True)
        print('greedy_set_process:', greedy_set_process, file=log_attack, flush=True)
        print('changed_set_process:', changed_set_process, file=log_attack, flush=True)
        print("  Number of query for this: " + str(query_num), file=log_attack, flush=True)
        print('greedy_set: ', file=log_attack, flush=True)
        print(greedy_set, file=log_attack, flush=True)
        print('greedy_funccall:', file=log_attack, flush=True)
        print(greedy_set_best_temp_funccall, file=log_attack, flush=True)
        print('best_prob = ' + str(g_process[-1]), file=log_attack, flush=True)
        print('best_object = ' + str(mf_process[-1]), file=log_attack, flush=True)
        print("  Number of changed codes: %d" % num_changed, file=log_attack, flush=True)
        print("risk funccall:", file=log_attack, flush=True)
        print('iteration: ' + str(iteration), file=log_attack, flush=True)
        print(" Time: " + str(all_t), file=log_attack, flush=True)
        if robust_flag == 0:
            print('flip_funccall:', file=log_attack, flush=True)
            print(flip_funccall, file=log_attack, flush=True)
            print('flip_set:', file=log_attack, flush=True)
            print(flip_set, file=log_attack, flush=True)
            print('flip_object = ', mf_process[-1], file=log_attack, flush=True)
            print(" The cardinality of S: " + str(len(greedy_set)), file=log_attack, flush=True)
        else:
            print(" The cardinality of S: " + str(len(greedy_set)) + ', but timeout', file=log_attack,
                  flush=True)
        print(" Adv acc: " + str(robust / (i + 1)), file=log_attack, flush=True)

        time_all.append(all_t)
        g_process_all.append(copy.deepcopy(g_process))
        mf_process_all.append(copy.deepcopy(mf_process))
        greedy_set_process_all.append(copy.deepcopy(greedy_set_process))
        changed_set_process_all.append(copy.deepcopy(changed_set_process))

        query_num_all.append(query_num)
        robust_flag_all.append(robust_flag)
        iteration_all.append(iteration)

        orignal_funccalls_all.append(copy.deepcopy(X[i].tolist()))
        orignal_labels_all.append(label)

        final_greedy_set_all.append(copy.deepcopy(greedy_set))
        final_funccall_all.append(copy.deepcopy(greedy_set_best_temp_funccall))
        final_changed_num_all.append(num_changed)

        if robust_flag == 0:
            flip_funccall_all.append(copy.deepcopy(flip_funccall))
            flip_set_all.append(copy.deepcopy(flip_set))
            flip_mf_all.append(mf_process[-1])
            flip_sample_original_label_all.append(label)
            flip_sample_index_all.append(i)

    else:
        final_funccall_all.append(copy.deepcopy(sample))

pickle.dump(changed_set_process_all,
            open(output_file + 'gradattack_changed_set_process_%d.pickle' % budget, 'wb'))
pickle.dump(g_process_all,
            open(output_file + 'gradattack_g_process_%d.pickle' % budget, 'wb'))
pickle.dump(mf_process_all,
            open(output_file + 'gradattack_mf_process_%d.pickle' % budget, 'wb'))
pickle.dump(greedy_set_process_all,
            open(output_file + 'gradattack_greedy_set_process_%d.pickle' % budget, 'wb'))
pickle.dump(query_num_all,
            open(output_file + 'gradattack_querynum_%d.pickle' % budget, 'wb'))
pickle.dump(robust_flag_all,
            open(output_file + 'gradattack_robust_flag_%d.pickle' % budget, 'wb'))
pickle.dump(orignal_funccalls_all,
            open(output_file + 'gradattack_original_funccall_%d.pickle' % budget, 'wb'))
pickle.dump(orignal_labels_all,
            open(output_file + 'gradattack_original_label_%d.pickle' % budget, 'wb'))
pickle.dump(final_greedy_set_all,
            open(output_file + 'gradattack_greedy_set_%d.pickle' % budget, 'wb'))
pickle.dump(final_funccall_all,
            open(output_file + 'gradattack_modified_funccall_%d.pickle' % budget, 'wb'))
pickle.dump(final_changed_num_all,
            open(output_file + 'gradattack_changed_num_%d.pickle' % budget, 'wb'))
pickle.dump(flip_funccall_all,
            open(output_file + 'gradattack_flip_funccall_%d.pickle' % budget, 'wb'))
pickle.dump(flip_set_all,
            open(output_file + 'gradattack_flip_set_%d.pickle' % budget, 'wb'))
pickle.dump(flip_mf_all,
            open(output_file + 'gradattack_flip_mf_%d.pickle' % budget, 'wb'))
pickle.dump(flip_sample_original_label_all,
            open(output_file + 'gradattack_flip_sample_original_label_%d.pickle' % budget, 'wb'))
pickle.dump(flip_sample_index_all,
            open(output_file + 'gradattack_flip_sample_index_%d.pickle' % budget, 'wb'))
pickle.dump(risk_funccall_all,
            open(output_file + 'gradattack_risk_funccall_%d.pickle' % budget, 'wb'))
pickle.dump(time_all,
            open(output_file + 'gradattack_time_%d.pickle' % budget, 'wb'))
pickle.dump(iteration_all,
            open(output_file + 'gradattack_iteration_%d.pickle' % budget, 'wb'))

write_file('IPS', 'Gradattack', budget, 'gradattack_', 3)