import sys
import pandas as pd
import numpy as np


def sum_normalization(x):
    x_row_sum = x.sum(axis=-1)
    x_row_sum = x_row_sum.reshape(list(x.shape)[:-1] + [1])
    return x / x_row_sum


print('data_dir=', sys.argv[1], 'k_1=', sys.argv[2], 'b=', sys.argv[3], 'k_2=', sys.argv[4])
data_dir = sys.argv[1]
train = pd.read_csv(data_dir + 'train_soft_label.csv', sep='\t', header=None)
use_cols = ['index', 'content', 'label_id', 'label']
train.columns = use_cols
train_contents_list = train['content'].tolist()

tf_dict_list = []
word_num_list = []
word_appear_label_time_dict = {}
for i in range(train['label_id'].nunique()):
    temp = train[train['label_id'] == i]
    temp = temp['content'].tolist()
    word_num = 0
    tf_dict_list.append({})
    for sentence in temp:
        sentence = sentence.split(' ')
        for word in sentence:
            if not tf_dict_list[i].get(word):
                tf_dict_list[i][word] = 0
            tf_dict_list[i][word] += 1
            word_num += 1
        sentence_set = set(sentence)
        for word in sentence_set:
            if not word_appear_label_time_dict.get(word):
                word_appear_label_time_dict[word] = {}
            word_appear_label_time_dict[word][i] = 1
    word_num_list.append(word_num)
for key in word_appear_label_time_dict:
    word_appear_label_time_dict[key] = len(word_appear_label_time_dict[key])

sentence_word_count_list = []
temp = train['content'].tolist()
word_appear_instance_time_dict = {}
for sentence in temp:
    sentence = sentence.split(' ')
    sentence_word_count_ = {}
    for word in sentence:
        if not sentence_word_count_.get(word):
            sentence_word_count_[word] = 0
        sentence_word_count_[word] += 1
    sentence_set = set(sentence)
    for word in sentence_set:
        if not word_appear_instance_time_dict.get(word):
            word_appear_instance_time_dict[word] = 0
        word_appear_instance_time_dict[word] += 1
    sentence_word_count_list.append(sentence_word_count_)

avg_dl = np.mean(word_num_list)
k_1 = float(sys.argv[2])
b = float(sys.argv[3])
k_2 = float(sys.argv[4])

label_size = train['label_id'].nunique()
temp = train[['content', 'label_id']]
wlr_dict_list = [{} for i in range(label_size)]
wlc_dict_list = [{} for i in range(label_size)]
wlw_dict = {}
k_list = []
for i in range(label_size):
    k_list.append(k_1 * (1 - b + b * word_num_list[i] / avg_dl))
for i in range(len(temp)):
    content_, label_id_ = temp.iloc[i]
    content_ = content_.split(' ')
    for word_ in content_:
        if not wlr_dict_list[label_id_].get(word_):
            f_ = tf_dict_list[label_id_][word_]
            k_ = k_list[label_id_]
            wlc = f_ * k_1 * 1. / (f_ + k_)
            wlw = wlw_dict.get(word_, np.log(label_size * 1. / word_appear_label_time_dict[word_]) + 1)
            wlr = wlc * wlw
            wlr_dict_list[label_id_][word_] = wlr
            wlc_dict_list[label_id_][word_] = wlc
            wlw_dict[word_] = wlw

temp = train['content'].tolist()
soft_label = []
N = len(temp)
wir_dict_list = []
wic_dict_list = []
wiw_dict = {}
x_avg = 0.
for index_, sentence in enumerate(temp):
    sentence = sentence.split(' ')
    x_avg += len(sentence)
    sentence = set(sentence)
    soft_label_ = [0] * label_size
    wir_dict_ = {}
    wic_dict_ = {}
    for word in sentence:
        if not wiw_dict.get(word):
            wiw = np.log(N * 1. / word_appear_instance_time_dict[word]) + 1
            wiw_dict[word] = wiw
        else:
            wiw = wiw_dict[word]
        f_ = sentence_word_count_list[index_][word]
        if not wir_dict_.get(word):
            wic = f_ * 1. * k_2 / (f_ + k_2)
            wir = wiw * wic
            wir_dict_[word] = wir
            wic_dict_[word] = wic
        else:
            wic = wic_dict_[word]
            wir = wir_dict_[word]
        for i in range(label_size):
            ilr = wlr_dict_list[i].get(word, 0) * wir
            soft_label_[i] += ilr
    soft_label_ = np.array(soft_label_)
    soft_label.append(soft_label_)
    wir_dict_list.append(wir_dict_)
    wic_dict_list.append(wic_dict_)
soft_label = np.array(soft_label)
x_avg /= len(temp)

soft_label = np.array(soft_label)
soft_label_norm = sum_normalization(np.exp(soft_label / x_avg))
np.save(data_dir + 'soft_label', soft_label_norm)
