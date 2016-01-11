# coding=utf-8
import sys

# ------------------------------------------------------ 归一化 ------------------------------------------------------
def get_max_min(train_pos_col, train_neg_col):
    tmp = []
    tmp.extend(train_pos_col)
    tmp.extend(train_neg_col)
    if len(tmp) == 0:
        print "preprocess.py - get_max_min() : train_column has no data."
        exit()
    max_value = max(tmp)
    min_value = min(tmp)
    if max_value <= min_value:
        print "preprocess.py - get_max_min() : max-{0} is equal or less to min-{1}.".format(max_value, min_value)
        exit()
    return max_value, min_value


def normalize(train_pos_column, train_neg_column, test_pos_column, test_neg_column):
    def do_normlize(max_value, min_value, column):
        deal_column = []
        for data in column:
            if data < min_value:
                data = min_value
            if data > max_value:
                data = max_value
            deal_column.append([(data - min_value) / (max_value - min_value)])
        return deal_column

    max_value, min_value = get_max_min(train_pos_column, train_neg_column)

    return do_normlize(max_value, min_value, train_pos_column), do_normlize(max_value, min_value, train_neg_column), \
           do_normlize(max_value, min_value, test_pos_column), do_normlize(max_value, min_value, test_neg_column), \
           max_value, min_value

# ------------------------------------------------------ 离散化 ------------------------------------------------------
def discrete_one_feature(bin, feature):
    position = -1
    discrete_feature = [0 for i in range(len(bin) + 1)]
    for idx in range(len(bin)):
        if bin[idx] > feature:
            position = idx
            break
    if position == -1:
        position = len(bin)
    discrete_feature[position] = 1
    return discrete_feature


def do_discrete(bin, feature_column):
    deal_column = []
    for feature in feature_column:
        deal_column.append(discrete_one_feature(bin, feature))
    return deal_column


def get_bin_freq(train_column, part):
    gap = len(train_column) / part
    feat_sort = sorted(train_column)
    bin = []
    index = 0
    num = 1
    while num < part:
        index += gap
        try:
            bin.append(feat_sort[index])
        except:
            bin.append(feat_sort[-1])
        num += 1
    return bin


def get_bin_value(train_column, part):
    max_value = max(train_column)
    min_value = min(train_column)
    interval = (max_value - min_value) / part
    boundary = min_value
    num = 1
    bin = []
    while num < part:
        boundary += interval
        bin.append(boundary)
        num += 1
    return bin


def discrete_freq(train_pos_column, train_neg_column, test_pos_column, test_neg_column, part):
    train_column = []
    train_column.extend(train_pos_column)
    train_column.extend(train_neg_column)
    bin = get_bin_freq(train_column, part)

    return do_discrete(bin, train_pos_column), do_discrete(bin, train_neg_column), \
           do_discrete(bin, test_pos_column), do_discrete(bin, test_neg_column), bin


def discrete_value(train_pos_column, train_neg_column, test_pos_column, test_neg_column, part):
    train_column = []
    train_column.extend(train_pos_column)
    train_column.extend(train_neg_column)
    bin = get_bin_value(train_column, part)

    return do_discrete(bin, train_pos_column), do_discrete(bin, train_neg_column), \
           do_discrete(bin, test_pos_column), do_discrete(bin, test_neg_column), bin


def discrete_bin(train_pos_column, train_neg_column, test_pos_column, test_neg_column, bin):
    return do_discrete(bin, train_pos_column), do_discrete(bin, train_neg_column), \
           do_discrete(bin, test_pos_column), do_discrete(bin, test_neg_column)

# ------------------------------------------------------ 离散化 - 特殊值 ------------------------------------------------------
def get_discrete_feature(special_fields, feature_column):
    '''
    筛选出非“特殊值”
    '''
    discrete_feature = []
    for feature in feature_column:
        if feature not in special_fields:
            discrete_feature.append(feature)
    return discrete_feature


def do_discrete_special(special_fields, bin, feature_column):
    '''
    做具有“特殊值”的离散化
    '''
    deal_column = []
    for feature in feature_column:
        if feature not in special_fields:
            special_feature = [0 for i in range(len(special_fields))]
            discrete_feature = discrete_one_feature(bin, feature)
        else:
            special_feature = [0 for i in range(len(special_fields))]
            index = special_fields.index(feature)
            special_feature[index] = 1
            discrete_feature = [0 for i in range(len(bin) + 1)]
        tmp = []
        tmp.extend(discrete_feature)
        tmp.extend(special_feature)
        deal_column.append(tmp)
    return deal_column


def discrete_special_freq(train_pos_column, train_neg_column, test_pos_column, test_neg_column, special_fields, part):
    discrete_feature_train_pos = get_discrete_feature(special_fields, train_pos_column)
    discrete_feature_train_neg = get_discrete_feature(special_fields, train_neg_column)
    discrete_feature_test_pos = get_discrete_feature(special_fields, test_pos_column)
    discrete_feature_test_neg = get_discrete_feature(special_fields, test_neg_column)
    # 对非“特殊值”进行离散化
    tmp1, tmp2, tmp3, tmp4, bin = discrete_freq(discrete_feature_train_pos, discrete_feature_train_neg,
                                                discrete_feature_test_pos, discrete_feature_test_neg, part)
    return do_discrete_special(special_fields, bin, train_pos_column), \
           do_discrete_special(special_fields, bin, train_neg_column), \
           do_discrete_special(special_fields, bin, test_pos_column), \
           do_discrete_special(special_fields, bin, test_neg_column), bin


def discrete_special_value(train_pos_column, train_neg_column, test_pos_column, test_neg_column, special_fields, part):
    discrete_feature_train_pos = get_discrete_feature(special_fields, train_pos_column)
    discrete_feature_train_neg = get_discrete_feature(special_fields, train_neg_column)
    discrete_feature_test_pos = get_discrete_feature(special_fields, test_pos_column)
    discrete_feature_test_neg = get_discrete_feature(special_fields, test_neg_column)
    # 对非“特殊值”进行离散化
    tmp1, tmp2, tmp3, tmp4, bin = discrete_value(discrete_feature_train_pos, discrete_feature_train_neg,
                                                 discrete_feature_test_pos, discrete_feature_test_neg, part)
    return do_discrete_special(special_fields, bin, train_pos_column), \
           do_discrete_special(special_fields, bin, train_neg_column), \
           do_discrete_special(special_fields, bin, test_pos_column), \
           do_discrete_special(special_fields, bin, test_neg_column), bin


def discrete_special_bin(train_pos_column, train_neg_column, test_pos_column, test_neg_column, special_fields, bin):
    return do_discrete_special(special_fields, bin, train_pos_column), \
           do_discrete_special(special_fields, bin, train_neg_column), \
           do_discrete_special(special_fields, bin, test_pos_column), \
           do_discrete_special(special_fields, bin, test_neg_column)

# ------------------------------------------------------ 分段 ------------------------------------------------------
def do_segmentation(interval, feature_column):
    '''
    将特征列按照 interval 分段
    '''
    interval_feature = [[] for i in range(len(interval) + 1)]
    for feature in feature_column:
        position = -1
        for idx in range(len(interval)):
            if interval[idx] > feature:
                position = idx
                break
        if position == -1:
            position = len(interval)
        interval_feature[position].append(feature)
    return interval_feature


def segmentation_freq(train_pos_column, train_neg_column, test_pos_column, test_neg_column, part):
    train_column = []
    train_column.extend(train_pos_column)
    train_column.extend(train_neg_column)
    interval = get_bin_freq(train_column, part)

    return do_segmentation(interval, train_pos_column), do_segmentation(interval, train_neg_column), \
           do_segmentation(interval, test_pos_column), do_segmentation(interval, test_neg_column), interval


def segmentation_value(train_pos_column, train_neg_column, test_pos_column, test_neg_column, part):
    train_column = []
    train_column.extend(train_pos_column)
    train_column.extend(train_neg_column)
    interval = get_bin_value(train_column, part)

    return do_segmentation(interval, train_pos_column), do_segmentation(interval, train_neg_column), \
           do_segmentation(interval, test_pos_column), do_segmentation(interval, test_neg_column), interval

def segmentation_bin(train_pos_column, train_neg_column, test_pos_column, test_neg_column, interval):

    return do_segmentation(interval, train_pos_column), do_segmentation(interval, train_neg_column), \
           do_segmentation(interval, test_pos_column), do_segmentation(interval, test_neg_column)


def segmentation_process(segmentation_process_sequence, dimension, interval, feature_column):
    deal_column = []
    for feature in feature_column:
        position = -1
        for idx in range(len(interval)):
            if interval[idx] > feature:
                position = idx
                break
        if position == -1:
            position = len(interval)
        strategy = segmentation_process_sequence[position][0]
        info = segmentation_process_sequence[position][1]

        deal_feature = []
        for i in range(position):
            deal_feature.extend([0 for j in range(dimension[i])])

        if strategy == "normalize":
            max_value = info[0]
            min_value = info[1]
            deal_feature.append((feature-min_value)/(max_value-min_value))
        else:
            bin = info
            deal_feature.extend(discrete_one_feature(bin, feature))

        for i in range(position + 1, len(interval) + 1):
            deal_feature.extend([0 for j in range(dimension[i])])

        deal_column.append(deal_feature)
    return deal_column
