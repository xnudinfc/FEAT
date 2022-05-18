import time
from itertools import combinations
import argparse
from utils import *
from model import *
import copy


# creating parser object
parser = argparse.ArgumentParser(description='FSGS')
parser.add_argument('--budget', default=5, type=int, help='purturb budget')
parser.add_argument('--dataset', default='IPS', type=str, help='dataset')
parser.add_argument('--modeltype', default='Normal', type=str, help='model type')
parser.add_argument('--time', default=60, type=int, help='time limit')
parser.add_argument('--t', default='True', type=str, help='test set or whole set')
args = parser.parse_args()

class Attacker(object):
    def __init__(self, best_parameters_file, log_f):
        # the classes of the dataset
        self.n_labels = num_classes[Dataset]
        self.model = RNN()

        if torch.cuda.is_available():
            self.model = self.model.cuda()
        # load the trained parameters of the classifier
        self.model.load_state_dict(torch.load(best_parameters_file, map_location='cpu'))
        # We only test data, so use this
        self.model.eval()
        # the log file
        self.log_f = log_f
        # loss function
        self.criterion = nn.CrossEntropyLoss()
        # the number of the categories of the dataset
        self.n_diagonosis_codes = num_category[Dataset]

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

    def eval_object(self, eval_funccall, current_object, greedy_set, orig_label, query_num):
        best_temp_funccall = copy.deepcopy(eval_funccall)
        candidate_lists = []
        success_flag = 1
        funccall_lists = []
        change_flag = 0
        worst_object = current_object
        # get the false labels
        label_set = set(range(self.n_labels))
        label_set.remove(orig_label)
        list_label_set = list(label_set)

        # first, classify the eval_funccall, if it is successful, skip all other things
        eval_pred, eval_g, eval_h = self.classify(eval_funccall, orig_label)
        query_num += 1
        object = eval_h - eval_g
        if object > 0:
            return object, eval_funccall, 0, query_num

        # update the worst object, which is the largest object
        if object >= worst_object:
            change_flag = 1
            worst_object = object
        # candidate_lists contains all the non-empty subsets of greedy_set that are within the budget
        if greedy_set:
            for i in range(1, min(len(greedy_set) + 1, budget)):
                subset1 = combinations(greedy_set, i)
                for subset in subset1:
                    candidate_lists.append(list(subset))

        # change the funccall based on the candidates above and get the candidate funccalls.
        for can in candidate_lists:
            temp_funccall = copy.deepcopy(eval_funccall)
            for position in can:
                visit_idx = position[0]
                code_idx = position[1]
                temp_funccall[visit_idx] = code_idx

            funccall_lists.append(temp_funccall)

        query_num += len(funccall_lists)

        # evaluate the candidate funccalls
        batch_size = 64
        n_batches = int(np.ceil(float(len(funccall_lists)) / float(batch_size)))
        for index in range(n_batches):  # n_batches

            batch_diagnosis_codes = funccall_lists[batch_size * index: batch_size * (index + 1)]
            batch_labels = [orig_label] * len(batch_diagnosis_codes)
            t_diagnosis_codes, t_labels = pad_matrix(batch_diagnosis_codes, batch_labels, self.n_diagonosis_codes)

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

            logit = self.model(model_input, t_diagnosis_codes)
            logit = logit.data.cpu().numpy()
            subsets_g = logit[:, orig_label]
            subsets_h = np.max([logit[:, false_class] for false_class in list_label_set], axis=0)
            subsets_object = subsets_h - subsets_g
            # get the maximum object, and update worst object
            max_object = np.max(subsets_object)
            max_index = np.argmax(subsets_object)

            if max_object >= worst_object:
                change_flag = 1
                worst_object = max_object
                best_temp_funccall = copy.deepcopy(funccall_lists[batch_size * index + max_index])

        # if change flag == 0, this means the worst object never changes, then we use the best funccall calculated
        # before as the solution for this part
        if change_flag == 0:
            success_flag = 2

        # attack successfully
        if worst_object > 0:
            success_flag = 0
            # print(worst_object)

        return worst_object, best_temp_funccall, success_flag, query_num

    # calculate which feature is changed
    def changed_set(self, eval_funccall, new_funccall):
        diff_set = set()
        for i in range(len(eval_funccall)):
            if eval_funccall[i] != new_funccall[i]:
                diff_set.add(i)
        return diff_set

    def attack(self, funccall, y):
        print()
        st = time.time()
        success_flag = 1
        orig_pred, orig_g, orig_h = self.classify(funccall, y)

        greedy_set = set()
        greedy_set_visit_idx = set()
        greedy_set_best_temp_funccall = funccall
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
        # when the classification is wrong for the original example, skip the attack
        if current_object > 0:
            robust_flag = -1
            print("Original classification error")

            return g_process, mf_process, greedy_set_process, changed_set_process, \
                   query_num, robust_flag, greedy_set, greedy_set_visit_idx, \
                   greedy_set_best_temp_funccall.tolist(),\
                   n_changed, flip_funccall.tolist(), flip_set, iteration

        print(current_object)
        # once the success_flag ==0, we attack successfully and exit
        while success_flag == 1:
            iteration += 1
            success_flag = 1
            candidate_objects = []
            candidate_funccalls = []
            candidate_poses = []

            # we loop over each feature and each category to find the worst object and its position
            for visit_idx in range(len(funccall)):
                if visit_idx in greedy_set_visit_idx:
                    continue
                worst_object_cate = -2
                best_pos_cate = -1
                best_temp_funccall_cate = funccall
                for code_idx in range(num_avail_category[Dataset]):

                    pos = (visit_idx, code_idx)
                    if code_idx == funccall[visit_idx]:
                        continue

                    eval_funccall = copy.deepcopy(funccall)
                    eval_funccall[visit_idx] = code_idx
                    worst_object, temp_funccall, success_flag_temp, query_num = self.eval_object(eval_funccall,
                                                                                                 current_object,
                                                                                                 greedy_set, y, query_num)

                    if success_flag_temp == 2:
                        temp_funccall = greedy_set_best_temp_funccall

                    if success_flag_temp == 0:
                        success_flag = 0

                    if worst_object > worst_object_cate:
                        worst_object_cate = worst_object
                        best_pos_cate = pos
                        best_temp_funccall_cate = temp_funccall

                candidate_objects.append(worst_object_cate)
                candidate_funccalls.append(best_temp_funccall_cate)
                candidate_poses.append(best_pos_cate)

            index = np.argmax(candidate_objects)
            max_object = np.max(candidate_objects)
            max_pos = candidate_poses[index]
            greedy_set_best_temp_funccall = candidate_funccalls[index]

            print(iteration)
            print('query', query_num)
            print(max_object)

            greedy_set.add(max_pos)
            greedy_set_visit_idx.add(max_pos[0])
            pred, g, h = self.classify(greedy_set_best_temp_funccall, y)
            g_process.append(np.float(g))
            mf_process.append(np.float(h - g))
            greedy_set_process.append(copy.deepcopy(greedy_set))
            if max_object > current_object:
                current_object = max_object
            changed_set_process.append(self.changed_set(funccall, greedy_set_best_temp_funccall))

            print(greedy_set)
            if success_flag == 1:
                if (time.time() - st) > time_limit:
                    success_flag = -1
                    robust_flag = 1

        n_changed = len(self.changed_set(funccall, greedy_set_best_temp_funccall))

        if robust_flag == 0:
            flip_funccall = greedy_set_best_temp_funccall
            flip_set = self.changed_set(funccall, flip_funccall)

        print("Modified_set:", flip_set)
        print(flip_funccall)

        return g_process, mf_process, greedy_set_process, changed_set_process, \
               query_num, robust_flag, greedy_set, greedy_set_visit_idx, \
               greedy_set_best_temp_funccall.tolist(), \
               n_changed, flip_funccall.tolist(), flip_set, iteration


Dataset = args.dataset
Model_Type = args.modeltype
budget = args.budget
time_limit = args.time
t = True
if args.t == 'False':
    t = False

print(Dataset, Model_Type)
output_file = './Logs/FSGS/'
make_dir(output_file)

X, y = load_data()
best_parameters_file = 'IPS_LSTM.par'

# ready for attack and log the files
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
final_changed_num_all = []
final_funccall_all = []

flip_funccall_all = []
flip_set_all = []
flip_mf_all = []
flip_sample_original_label_all = []
flip_sample_index_all = []

iteration_all = []
time_all = []

log_attack = open(
    './Logs/FSGS/greedmax_Attack.bak', 'w+')
attacker = Attacker(best_parameters_file, log_attack)

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
    greedy_set, greedy_set_visit_idx, greedy_set_best_temp_funccall, \
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

    if robust_flag != -1:
        print('g_process:', g_process, file=log_attack, flush=True)
        print('mf_process:', mf_process, file=log_attack, flush=True)
        print('greedy_set_process:', greedy_set_process, file=log_attack, flush=True)
        print('changed_set_process:', changed_set_process, file=log_attack, flush=True)
        print("  Number of query for this: " + str(query_num), file=log_attack, flush=True)
        print('greedy_set: ', file=log_attack, flush=True)
        print(greedy_set, file=log_attack, flush=True)
        print('greedy_set_visit_idx: ', file=log_attack, flush=True)
        print(greedy_set_visit_idx, file=log_attack, flush=True)
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
        final_greedy_set_visit_idx_all.append(copy.deepcopy(greedy_set_visit_idx))
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

pickle.dump(g_process_all,
            open(output_file + 'greedmax_g_process_%d.pickle' % budget, 'wb'))
pickle.dump(mf_process_all,
            open(output_file + 'greedmax_mf_process_%d.pickle' % budget, 'wb'))
pickle.dump(greedy_set_process_all,
            open(output_file + 'greedmax_greedy_set_process_%d.pickle' % budget, 'wb'))
pickle.dump(changed_set_process_all,
            open(output_file + 'greedmax_changed_set_process_%d.pickle' % budget, 'wb'))
pickle.dump(query_num_all,
            open(output_file + 'greedmax_querynum_%d.pickle' % budget, 'wb'))
pickle.dump(robust_flag_all,
            open(output_file + 'greedmax_robust_flag_%d.pickle' % budget, 'wb'))
pickle.dump(orignal_funccalls_all,
            open(output_file + 'greedmax_original_funccall_%d.pickle' % budget, 'wb'))
pickle.dump(orignal_labels_all,
            open(output_file + 'greedmax_original_label_%d.pickle' % budget, 'wb'))
pickle.dump(final_greedy_set_all,
            open(output_file + 'greedmax_greedy_set_%d.pickle' % budget, 'wb'))
pickle.dump(final_greedy_set_visit_idx_all,
            open(output_file + 'greedmax_feature_greedy_set_%d.pickle' % budget, 'wb'))
pickle.dump(final_changed_num_all,
            open(output_file + 'greedmax_changed_num_%d.pickle' % budget, 'wb'))
pickle.dump(final_funccall_all,
            open(output_file + 'greedmax_modified_funccall_%d.pickle' % budget, 'wb'))
pickle.dump(flip_funccall_all,
            open(output_file + 'greedmax_flip_funccall_%d.pickle' % budget, 'wb'))
pickle.dump(flip_set_all,
            open(output_file + 'greedmax_flip_set_%d.pickle' % budget, 'wb'))
pickle.dump(flip_mf_all,
            open(output_file + 'greedmax_flip_mf_%d.pickle' % budget, 'wb'))
pickle.dump(flip_sample_original_label_all,
            open(output_file + 'greedmax_flip_sample_original_label_%d.pickle' % budget, 'wb'))
pickle.dump(flip_sample_index_all,
            open(output_file + 'greedmax_flip_sample_index_%d.pickle' % budget, 'wb'))
pickle.dump(iteration_all,
            open(output_file + 'greedmax_iteration_%d.pickle' % budget, 'wb'))
pickle.dump(time_all,
            open(output_file + 'greedmax_time_%d.pickle' % budget, 'wb'))

write_file(Dataset, Model_Type, budget, 'greedmax_', time_limit)

