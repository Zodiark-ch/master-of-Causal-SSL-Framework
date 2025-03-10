#!/usr/bin/python2.7


import numpy as np
import argparse
import torch


def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content


def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)

    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default="gtea")
    parser.add_argument('--split', default='4')

    args = parser.parse_args()

    ground_truth_path = "/home/data/dukee/asrf-main/dataset/"+args.dataset+"/groundTruth/"
    recog_path = "./results/"+args.dataset+"/split_"+args.split+"/"
    file_list = "/home/data/dukee/asrf-main/dataset/"+args.dataset+"/splits/test.split"+args.split+".bundle"
    
    # 加一个mapping path
    mapping_file = "/home/data/dukee/asrf-main/dataset/"+args.dataset+"/mapping.txt"
    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    num_classes = len(actions_dict)

    list_of_videos = read_file(file_list).split('\n')[:-1]

    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

    correct = 0
    total = 0
    edit = 0
    hamming = 0
    
    for vid in list_of_videos:
        gt_target = []
        recog_target = []
        
        
        gt_file = ground_truth_path + vid
        gt_content = read_file(gt_file).split('\n')[0:-1]
        gt_class = np.zeros(len(gt_content))      
        for i in range(len(gt_class)):
            gt_class[i] = actions_dict[gt_content[i]]
        gt_target.append(gt_class)
        gt_target_tensor = torch.ones(len(gt_content), dtype=torch.long)*(-100)
        gt_target_tensor[:np.shape(gt_target[0])[0]] = torch.from_numpy(gt_target[0])  # torch.Size([11679])
        #--------------------------------------------------------
        exp_gt_target_tensor = gt_target_tensor.unsqueeze(0)
        shape = exp_gt_target_tensor.shape
        broad_exp_gt_i = exp_gt_target_tensor.expand(shape[1],shape[1])
        broad_exp_gt_j = torch.transpose(broad_exp_gt_i, 0, 1)

        exp_gt = torch.ne(broad_exp_gt_i, broad_exp_gt_j).to(torch.int)
        #==============================================================================================
        recog_file = recog_path + vid.split('.')[0]
        recog_content = read_file(recog_file).split('\n')[1].split()
        
        recog_class = np.zeros(len(gt_content)) 
        for i in range(len(gt_class)):
            recog_class[i] = actions_dict[recog_content[i]]
        recog_target.append(recog_class)
        recog_target_tensor = torch.ones(len(gt_content), dtype=torch.long)*(-100)
        recog_target_tensor[:np.shape(recog_target[0])[0]] = torch.from_numpy(recog_target[0])
        #--------------------------------------------------------
        exp_recog_target_tensor = recog_target_tensor.unsqueeze(0)
        shape = exp_recog_target_tensor.shape
        broad_exp_recog_i = exp_recog_target_tensor.expand(shape[1],shape[1])
        broad_exp_recog_j = torch.transpose(broad_exp_recog_i, 0, 1)

        exp_recog = torch.ne(broad_exp_recog_i, broad_exp_recog_j).to(torch.int)
        
        differences = torch.sum(exp_gt != exp_recog)
        hamming += differences.data/(shape[1]*shape[1])

        for i in range(len(gt_content)):
            total += 1
            if gt_content[i] == recog_content[i]:
                correct += 1

        edit += edit_score(recog_content, gt_content)

        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1

    print("Acc: %.4f" % (100*float(correct)/total))
    print('Edit: %.4f' % ((1.0*edit)/len(list_of_videos)))
    print('Hamming: %.4f' % hamming)
    acc = (100*float(correct)/total)
    edit = ((1.0*edit)/len(list_of_videos))
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s]+fp[s])
        recall = tp[s] / float(tp[s]+fn[s])

        f1 = 2.0 * (precision*recall) / (precision+recall)

        f1 = np.nan_to_num(f1)*100
        print('F1@%0.2f: %.4f' % (overlap[s], f1))

if __name__ == '__main__':
    main()
