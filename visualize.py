# This code is part of the Lip-reading project for 670 F2020
# to visualize results.
# by Catherine Huang

import numpy as np
import matplotlib.pyplot as plt
from facial_landmarks import get_all_files_path

def get_ratio_per_class(pred, true, num_class):
    pos_neg_array = []
    for i in range(num_class):
        # for each class
        # get the idx for all the 
        # print("true min: {} max: {}".format(np.min(true), np.max(true)))
        true_idx= np.where(true==i)
        true_array = true[true_idx]
        pred_array = pred[true_idx]
        compare = np.equal(pred_array, true_array)
        num_true = compare.sum()
        # print(true_idx)
        num_false = true_array.shape[0] - num_true
        # print("total: {} true: {} false: {}".format((true_array.shape[0]), num_true, num_false))
        pos_neg_array.append([num_true, num_false])
    return np.array(pos_neg_array)

def bar_graph(ratio):
    truth = ratio[:30,0]
    false = ratio[:30,1]
    ind = np.arange(30)    # the x locations for the groups
    width = 0.3       # the width of the bars: can also be len(x) sequence

    fig, ax = plt.subplots()
    
    rects1 = ax.bar(ind - width/2, truth, width,
                    label='Correct Prediction')
    rects2 = ax.bar(ind + width/2, false, width,
                    label='Incorrect Prediction')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    # ax.set_yticks(np.arange(0, 30, 5))
    ax.set_title('Scores by prediction per class')
    ax.set_xticks(ind)
    ax.set_xticklabels(classes[:30], rotation = 45, ha="right")
    ax.legend()
    
    fig.set_size_inches(30, 45)
    fig.tight_layout()

    plt.savefig("prediction_first_30.png")

# TODO: refactor, don't need 3 of these
def bar_graph_max(ratio):
    # print("ratio shape {}".format(ratio.shape))
    # pritn()
    max_idx, _ = np.where(ratio == np.max(ratio[:,0]))
    print("max idx : {} shape {}".format(max_idx, len(max_idx)))
    truth = ratio[max_idx,0]
    false = ratio[max_idx,1]
    print("truth shape: {}".format(truth.shape))
    ind = np.arange(len(max_idx))    # the x locations for the groups
    width = 0.3       # the width of the bars: can also be len(x) sequence
    classes_np = np.array(classes)
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind - width/2, truth, width, label='Correct Prediction')
    rects2 = ax.bar(ind + width/2, false, width, label='Incorrect Prediction')
    max_idx = np.array(max_idx)
    print(max_idx)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('classes with highest correct prediction')
    ax.set_xticks(ind)
    ax.set_xticklabels(classes_np[max_idx], rotation = 45, ha="right")
    ax.legend()

    fig.tight_layout()

    plt.savefig("highest_correct.png")

def bar_graph_min(ratio):
    # print("ratio shape {}".format(ratio.shape))
    # pritn()
    mix_idx, _ = np.where(ratio == np.min(ratio[:,0]))
    print("max idx : {} shape {}".format(mix_idx, len(mix_idx)))
    truth = ratio[mix_idx,0]
    false = ratio[mix_idx,1]
    print("truth shape: {}".format(truth.shape))
    ind = np.arange(len(mix_idx))    # the x locations for the groups
    width = 0.3       # the width of the bars: can also be len(x) sequence
    classes_np = np.array(classes)
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind - width/2, truth, width, label='Correct Prediction')
    rects2 = ax.bar(ind + width/2, false, width, label='Incorrect Prediction')
    mix_idx = np.array(mix_idx)
    print(mix_idx)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('classes with highest incorrect prediction')
    ax.set_xticks(ind)
    ax.set_xticklabels(classes_np[mix_idx], rotation = 45, ha="right")
    ax.legend()

    fig.tight_layout()

    plt.savefig("highest_incorrect.png")

# display results
classes, files = get_all_files_path()

tp_load = np.load('for_truth_tables.npz')
# print(tp_load.files)
pred = tp_load['pred']
truth = tp_load['truth']
# print(pred.shape)
# # print(tp_load)
# # print("classes: {}".format(classes))
class_ratio = get_ratio_per_class(pred, truth, len(classes))
# # print(get_ratio_per_class(pred, truth, len(classes)))
# print(class_ratio.shape)

# draw and save the graphs
bar_graph(class_ratio)
bar_graph_max(class_ratio)
bar_graph_min(class_ratio)

# TODO: add code to do truth table and accuracy, percision, recall scores per class level
