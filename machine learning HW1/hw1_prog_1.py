import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

MNIST_IMG_SIZE = 28

def display_digit(input_array):
    input_array = np.array(input_array)
    im = input_array.reshape(28, 28) # assume digits are 28*28 grayscale bitmap
    plt.gray()
    plt.imshow(im)
    plt.show()

def display_each_digit_once(img_set, n_row, n_col):
    for i in xrange(len(img_set)):
        plt.subplot(n_row, n_col, i + 1)
        plt.gray()
        plt.imshow(np.array(img_set[i]).reshape(MNIST_IMG_SIZE, MNIST_IMG_SIZE))
    plt.show()

def get_each_digit_sample(file_name):
    count = 0
    digit = [False] * 10
    digit_img = [[]] * 10

    # Find out one feature vector for each digit
    #
    # No exception handlers for digits not existing in the dataset, that is, we
    # assume we can find samples for each digit
    with open(file_name, 'r') as f:
        for line in f:
            count += 1
            if count > 1: # skip the first line
                line = line.strip()
                buf = [int(i) for i in line.split(',')]
                digit[buf[0]] = True
                digit_img[buf[0]] = buf[1:]
            if sum(digit) == 10:
                break
    return digit_img

def get_zero_and_one_sample(file_name):
    count = 0
    result_set = [[] for _ in xrange(2)] # ref: http://stackoverflow.com/questions/
                                         # 8713620/appending-items-to-a-list-of-lists
                                         # -in-python
    with open(file_name, 'r') as f:
        for line in f:
            count += 1
            line = line.strip()
            if count > 1:
                buf = [int(num) for num in line.split(',')]
                if buf[0] == 0:
                    result_set[0].append(buf[1:])
                elif buf[0] == 1:
                    result_set[1].append(buf[1:])
    return result_set

def plot_histogram(genuine_dis, imposter_dis):
        max_val = max(genuine_dis.tolist() + imposter_dis.tolist())
        min_val = min(genuine_dis.tolist() + imposter_dis.tolist())
        num_tick = 100
        tick_label = np.linspace(min_val, max_val, num_tick + 1)[1:]
        tick_label = ['{0:.2f}'.format(tick) for tick in tick_label]
        hist1, bin_edge_1 = np.histogram(genuine_dis, np.linspace(min_val, max_val
                                         , num_tick + 1))
        hist2, bin_edge_2 = np.histogram(imposter_dis, np.linspace(min_val, max_val
                                         , num_tick + 1))
        bar_width = 0.4
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.bar(range(len(bin_edge_1) - 1), hist1.tolist(), bar_width, color='b'
                , label='Genuine')
        plt.bar(np.array(range(len(bin_edge_2) - 1)) + 0.4, hist2.tolist(), bar_width, color='r'
                , label='Imposter')
        ax.set_xticks(np.array([0, 20, 40, 60, 80, 100]))
        ax.set_xticklabels([tick_label[0], tick_label[19], tick_label[39], tick_label[59]
                           , tick_label[79], tick_label[99]])
        plt.xlabel('Distance')
        plt.ylabel('# Pair')
        plt.title('Distance Histogram for Genuine and Imposter Pairs of 0 and 1 samples')
        plt.legend()
        plt.show()

def plot_roc_curve(genuine_dis, imposter_dis):
        max_val = max(genuine_dis.tolist() + imposter_dis.tolist())
        min_val = min(genuine_dis.tolist() + imposter_dis.tolist())
        num_tick = 100
        hist_genuine, bin_edge_1 = np.histogram(genuine_dis, np.linspace(min_val, max_val
                                         , num_tick + 1))
        hist_imposter, bin_edge_2 = np.histogram(imposter_dis, np.linspace(min_val, max_val
                                         , num_tick + 1))
        sum_hist_genuine = sum(hist_genuine)
        sum_hist_imposter = sum(hist_imposter)
        roc_point_x = []
        roc_point_y = []
        i = 0
        for i in xrange(num_tick):
            if hist_imposter[i] != 0:
                break
        i -= 1
        if i < 0:
            i += 1
        acc_imposter = float(hist_imposter[i])
        acc_genuine = float(sum(hist_genuine[:i + 1]))
        while True:
            fpr = acc_imposter / sum_hist_imposter
            tpr = acc_genuine / sum_hist_genuine
            roc_point_x.append(fpr)
            roc_point_y.append(tpr)
            i += 1
            if i == num_tick:
                break
            else:
                acc_imposter += hist_imposter[i]
                acc_genuine += hist_genuine[i]
        plt.plot(roc_point_x, roc_point_y, 'k', [1, 0], [0, 1], 'b--')
        plt.plot([0.2145], [1 - 0.2145], 'ro') # the EER point
        plt.text(0.213, 0.81, 'EER')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for genuine and imposter pairs of 0 and 1 samples')
        plt.show()

def load_dataset(file_name):
    label = []
    feature = []
    count = 0
    #debug_limit = 6
    with open(file_name, 'r') as f:
        for line in f:
            count += 1
            if count > 1:
                line = line.strip()
                buf = line.split(',')
                buf = [int(i) for i in buf]
                label.append(buf[0])
                feature.append(buf[1:])
            #if count == debug_limit:
            #    break
    print 'Data loaded successfully'
    return label, feature

def get_boundary(num, bins):
    result = []
    bin_size = num / bins
    lower = 0
    for i in xrange(bins):
        if i < bins - 1:
            result.append([lower, lower + bin_size])
            lower += bin_size
        else:
            result.append([lower, num])
            break
    return result

def knn_classifier(label, feature, k):
    #pair_dis = cdist(feature, feature)
    size = len(feature)  
    #print pair_dis
    hit = 0
    for i in xrange(size):
        #dis_list = []
        #for j in xrange(size):
        #    if i != j:
        #        dis_list.append([j, pair_dis[i, j]])
        dis_pool = cdist([feature[i]], feature[:i] + feature[i + 1:]).tolist()[0]
        #print dis_pool
        index_set = range(i) + range(i + 1, size)
        dis_list = []
        for idx, item in enumerate(dis_pool):
            dis_list.append([index_set[idx], item])
        #print dis_list
        dis_list = sorted(dis_list, key=lambda x: x[1])
        #print dis_list
        #print dis_list[:k]
        #print label
        predict_label = np.bincount([label[item[0]] for item in dis_list[:k]]).argmax()
        #print predict_label
        if predict_label == label[i]:
            hit += 1
        print 'Up to sample {0}, {1} hits'.format(i, hit)
    return float(hit) / size

def count_digit_freq(file_name):
    # 1. X-axis ticks need to be improved, only 2 4 6 8 10 here
    count = 0
    digit_freq = [0] * 10
    with open(file_name, 'r') as f:
        for line in f:
            count += 1
            if count > 1:
                digit = int(line.strip().split(',')[0])
                digit_freq[digit] += 1
    freq_sum = sum(digit_freq)
    norm_digit_freq = [float(freq) / freq_sum for freq in digit_freq]
    plt.title('Normalized Frequencies of MNIST Digits')
    plt.xlabel('Digit')
    plt.ylabel('Normalized Frequency')
    plt.bar(range(10), norm_digit_freq)
    plt.show()

def find_most_similar_sample_2norm(dataset, query_set):
    response_set = [[]] * 10
    count = 0
    with open(dataset, 'r') as f:
        for line in f:
            count += 1
            if count > 1:
                buf = [int(i) for i in line.strip().split(',')]
                if count == 2:
                    for i in xrange(10):
                        response_set[i] = buf
                elif count > 2:
                    for i in xrange(10):
                        if (np.linalg.norm(np.array(query_set[i]) - np.array(buf[1:]))
                           < np.linalg.norm(np.array(query_set[i]) - np.array(response_set[i][1:]))
                           and query_set[i] != buf[1:]):
                            response_set[i] = buf
    return response_set

def compute_pair_dis(set1, set2, mode):
    result = []
    if mode == 'genuine':
        result = np.unique(np.ravel(cdist(set1, set1)))
    elif mode == 'imposter':
        result = np.ravel(cdist(set1, set2))
    return result

def main(dataset, prob_id):
    if prob_id == 'b':
        img_set = get_each_digit_sample(dataset)
        display_each_digit_once(img_set, 2, 5)
    elif prob_id == 'c':
        count_digit_freq(dataset)
    elif prob_id == 'd':
        query_set = get_each_digit_sample(dataset)
        response_set = find_most_similar_sample_2norm(dataset, query_set)
        for idx, item in enumerate(response_set):
            response_set[idx] = item[1:]
        display_each_digit_once(response_set, 2, 5)
    elif prob_id == 'e':
        zero_one_set = get_zero_and_one_sample(dataset)
        print len(zero_one_set[0])
        print len(zero_one_set[1])
        genuine_dis = compute_pair_dis(zero_one_set[0], zero_one_set[0], 'genuine')
        genuine_dis = np.hstack((genuine_dis
                                , compute_pair_dis(zero_one_set[1], zero_one_set[1], 'genuine')))
        imposter_dis = compute_pair_dis(zero_one_set[0], zero_one_set[1], 'imposter')
        plot_histogram(genuine_dis, imposter_dis)
    elif prob_id == 'f':
        zero_one_set = get_zero_and_one_sample(dataset)
        genuine_dis = compute_pair_dis(zero_one_set[0], zero_one_set[0], 'genuine')
        genuine_dis = np.hstack((genuine_dis
                                , compute_pair_dis(zero_one_set[1], zero_one_set[1], 'genuine')))
        imposter_dis = compute_pair_dis(zero_one_set[0], zero_one_set[1], 'imposter')
        plot_roc_curve(genuine_dis, imposter_dis)
    elif prob_id == 'h':
        label, feature = load_dataset(dataset)
        label_set = label[:28000]
        feature_set = feature[:28000]
        print knn_classifier(label_set, feature_set, 3)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
