__author__ = 'BFD_308'
# coding: UTF-8
import os
import json
import sys

import lr
import preprocess
import save_model_lr


def do_mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    else:
        pass

def remove_dir(dir_path):
    if os.path.exists(dir_path):
        os.rmdir(dir_path)
    else:
        pass


def load_matrix(feature_file):
    matrix = []
    with open(feature_file) as f:
        for line in f:
            chunks = line.strip().split("\t")
            row = []
            for chunk in chunks:
                try:
                    row.append(float(chunk))
                except:
                    print "offline_exp.py - load_matrix() : feature-{0} is not a valid number in file-{1].".format(
                        chunk, feature_file)
                    exit()
            matrix.append(row)
    return matrix


def record_process_info(result_dir, idx, feature, info_list):
    writer = open(result_dir + "{0}-{1}.process".format(idx, feature), "w")
    for info in info_list:
        writer.write(info)
        writer.write("\n")
    writer.close()


def record_feature_matrix(feature_matrix, output_file):
    writer = open(output_file, "w")
    for feature1 in feature_matrix:
        feature2 = [str(one) for one in feature1]
        writer.write("\t".join(feature2))
        writer.write("\n")
    writer.close()

class PreProcess:
    def __init__(self, train_pos, train_neg, test_pos, test_neg, result_dir, feature_info_dict):
        self.X_train_pos = load_matrix(train_pos)
        self.X_train_neg = load_matrix(train_neg)
        self.X_test_pos = load_matrix(test_pos)
        self.X_test_neg = load_matrix(test_neg)
        self.result_dir = result_dir
        self.feature_process_dict = feature_info_dict


    def get_column(self, index):
        if index >= len(self.X_train_pos[0]):
            print "offline_exp.py - get_column() : index-{0} out of scope-{1}".format(index, len(self.X_train_pos[0]))
            exit()
        if index >= len(self.X_train_neg[0]):
            print "offline_exp.py - get_column() : index-{0} out of scope-{1}".format(index, len(self.X_train_neg[0]))
            exit()
        if index >= len(self.X_test_pos[0]):
            print "offline_exp.py - get_column() : index-{0} out of scope-{1}".format(index, len(self.X_test_pos[0]))
            exit()
        if index >= len(self.X_test_neg[0]):
            print "offline_exp.py - get_column() : index-{0} out of scope-{1}".format(index, len(self.X_test_neg[0]))
            exit()

        train_pos_col = [one[index] for one in self.X_train_pos]
        train_neg_col = [one[index] for one in self.X_train_neg]
        test_pos_col = [one[index] for one in self.X_test_pos]
        test_neg_col = [one[index] for one in self.X_test_neg]

        return train_pos_col, train_neg_col, test_pos_col, test_neg_col

    def do_normalize(self, train_pos_col, train_neg_col, test_pos_col, test_neg_col):
        train_pos_col_norm, train_neg_col_norm, test_pos_col_norm, test_neg_col_norm, max_value, min_value = \
            preprocess.normalize(train_pos_col, train_neg_col, test_pos_col, test_neg_col)

        return train_pos_col_norm, train_neg_col_norm, test_pos_col_norm, test_neg_col_norm, max_value, min_value

    def do_discrete_freq(self, strategy, train_pos_col, train_neg_col, test_pos_col, test_neg_col):
        part = int(strategy.split(":")[1])
        train_pos_col_dis, train_neg_col_dis, test_pos_col_dis, test_neg_col_dis, bin = \
            preprocess.discrete_freq(train_pos_col, train_neg_col, test_pos_col, test_neg_col, part)

        return train_pos_col_dis, train_neg_col_dis, test_pos_col_dis, test_neg_col_dis, bin

    def do_discrete_value(self, strategy, train_pos_col, train_neg_col, test_pos_col, test_neg_col):
        part = int(strategy.split(":")[1])
        train_pos_col_dis, train_neg_col_dis, test_pos_col_dis, test_neg_col_dis, bin = \
            preprocess.discrete_value(train_pos_col, train_neg_col, test_pos_col, test_neg_col, part)

        return train_pos_col_dis, train_neg_col_dis, test_pos_col_dis, test_neg_col_dis, bin

    def do_discrete_bin(self, strategy, train_pos_col, train_neg_col, test_pos_col, test_neg_col):
        boundary_list = strategy.split(":")[1].split("-")
        bin = [float(one) for one in boundary_list]
        train_pos_col_dis, train_neg_col_dis, test_pos_col_dis, test_neg_col_dis = \
            preprocess.discrete_bin(train_pos_col, train_neg_col, test_pos_col, test_neg_col, bin)

        return train_pos_col_dis, train_neg_col_dis, test_pos_col_dis, test_neg_col_dis, bin


    def do_process(self, idx, feature, process_info):
        '''
        对一列特征进行处理
        根据 process_info 调用函数，进行串行的特征处理
        '''
        print idx
        print feature
        print process_info
        print

        process_info_result = []
        train_pos_col, train_neg_col, test_pos_col, test_neg_col = self.get_column(idx)
        strategy_list = process_info.split(";")
        strategy_index = 0
        strategy = strategy_list[strategy_index].strip()
        strategy_index += 1
        if strategy.startswith("normalize"):
            rt1, rt2, rt3, rt4, max_value, min_value = self.do_normalize(train_pos_col, train_neg_col, test_pos_col, test_neg_col)
            # 记录 max, min
            process_info_result.append("normalize : " + json.dumps({"max": max_value, "min": min_value}))
            return rt1, rt2, rt3, rt4, process_info_result

        elif strategy.startswith("discrete_freq"):
            rt1, rt2, rt3, rt4, bin = self.do_discrete_freq(strategy, train_pos_col, train_neg_col, test_pos_col, test_neg_col)
            # 记录 bin
            process_info_result.append("discrete_freq : " + json.dumps({"bin": "|".join([str(one) for one in bin])}))
            return rt1, rt2, rt3, rt4, process_info_result

        elif strategy.startswith("discrete_value"):
            rt1, rt2, rt3, rt4, bin = self.do_discrete_value(strategy, train_pos_col, train_neg_col, test_pos_col, test_neg_col)
            # 记录 bin
            process_info_result.append("discrete_value : " + json.dumps({"bin": "|".join([str(one) for one in bin])}))
            return rt1, rt2, rt3, rt4, process_info_result

        elif strategy.startswith("discrete_bin"):
            rt1, rt2, rt3, rt4, bin = self.do_discrete_bin(strategy, train_pos_col, train_neg_col, test_pos_col, test_neg_col)
            # 记录 bin
            process_info_result.append("discrete_bin : " + json.dumps({"bin": "|".join([str(one) for one in bin])}))
            return rt1, rt2, rt3, rt4, process_info_result

        elif strategy.startswith("discrete_special_freq"):
            special_list = strategy.split(":")[1].split(" ")[0].split("-")
            special_list = [float(one) for one in special_list]
            part = int(strategy.split(":")[1].split(" ")[1])
            train_pos_col_dis, train_neg_col_dis, test_pos_col_dis, test_neg_col_dis, bin = preprocess.discrete_special_freq\
                (train_pos_col, train_neg_col, test_pos_col, test_neg_col, special_list, part)
            # 记录 bin
            process_info_result.append("discrete_special_freq : " + json.dumps({"bin": "|".join([str(one) for one in bin])}))
            return train_pos_col_dis, train_neg_col_dis, test_pos_col_dis, test_neg_col_dis, process_info_result

        elif strategy.startswith("discrete_special_value"):
            special_list = strategy.split(":")[1].split(" ")[0].split("-")
            special_list = [float(one) for one in special_list]
            part = int(strategy.split(":")[1].split(" ")[1])
            train_pos_col_dis, train_neg_col_dis, test_pos_col_dis, test_neg_col_dis, bin = preprocess.discrete_special_value\
                (train_pos_col, train_neg_col, test_pos_col, test_neg_col, special_list, part)
            # 记录 bin
            process_info_result.append("discrete_special_value : " + json.dumps({"bin": "|".join([str(one) for one in bin])}))
            return train_pos_col_dis, train_neg_col_dis, test_pos_col_dis, test_neg_col_dis, process_info_result

        elif strategy.startswith("discrete_special_bin"):
            special_list = strategy.split(":")[1].split(" ")[0].split("-")
            special_list = [float(one) for one in special_list]
            boundary_list = strategy.split(":")[1].split(" ")[1].split("-")
            bin = [float(one) for one in boundary_list]
            train_pos_col_dis, train_neg_col_dis, test_pos_col_dis, test_neg_col_dis = preprocess.discrete_special_bin\
                (train_pos_col, train_neg_col, test_pos_col, test_neg_col, special_list, bin)
            # 记录 bin
            process_info_result.append("discrete_special_bin : " + json.dumps({"bin": "|".join([str(one) for one in bin])}))
            return train_pos_col_dis, train_neg_col_dis, test_pos_col_dis, test_neg_col_dis, process_info_result


        elif strategy.startswith("segmentation"):

            if strategy.startswith("segmentation_freq"):
                part = int(strategy.split(":")[1])
                train_pos_col_interval, train_neg_col_interval, test_pos_col_interval, test_neg_col_interval, interval = \
                    preprocess.segmentation_freq(train_pos_col, train_neg_col, test_pos_col, test_neg_col, part)
                process_info_result.append("segmentation_freq : " + json.dumps({"interval": "|".join([str(one) for one in interval])}))

            elif strategy.startswith("segmentation_value"):
                part = int(strategy.split(":")[1])
                train_pos_col_interval, train_neg_col_interval, test_pos_col_interval, test_neg_col_interval, interval = \
                    preprocess.segmentation_value(train_pos_col, train_neg_col, test_pos_col, test_neg_col, part)
                process_info_result.append("segmentation_value : " + json.dumps({"interval": "|".join([str(one) for one in interval])}))

            elif strategy.startswith("segmentation_bin"):
                boundary_list = strategy.split(":")[1].split("-")
                interval = [float(one) for one in boundary_list]
                part = len(interval) + 1
                train_pos_col_interval, train_neg_col_interval, test_pos_col_interval, test_neg_col_interval = \
                    preprocess.segmentation_bin(train_pos_col, train_neg_col, test_pos_col, test_neg_col, interval)
                process_info_result.append("segmentation_bin : " + json.dumps({"interval": "|".join([str(one) for one in interval])}))

            else:
                print "offline_exp.py - do_process() : strategy-{0} error.".format(strategy)
                exit()

            segmentation_process_sequence = []
            dimension = []
            for i in range(part):
                strategy = strategy_list[strategy_index].strip()
                strategy_index += 1
                if strategy.startswith("normalize"):
                    dimension.append(1)
                    max_value, min_value = preprocess.get_max_min(train_pos_col_interval[i], train_neg_col_interval[i])
                    segmentation_process_sequence.append(["normalize", [max_value, min_value]])
                    # 记录 max, min
                    process_info_result.append("interval-{0} normalize : ".format(i) + json.dumps({"max": max_value, "min": min_value}))

                elif strategy.startswith("discrete_freq"):
                    part = int(strategy.split(":")[1])
                    dimension.append(part)
                    tmp = []
                    tmp.extend(train_pos_col_interval[i])
                    tmp.extend(train_neg_col_interval[i])
                    bin = preprocess.get_bin_freq(tmp, part)
                    segmentation_process_sequence.append(["discrete", bin])
                    # 记录 bin
                    process_info_result.append("interval-{0} discrete_freq : ".format(i) + json.dumps({"bin": "|".join([str(one) for one in bin])}))

                elif strategy.startswith("discrete_value"):
                    part = int(strategy.split(":")[1])
                    dimension.append(part)
                    tmp = []
                    tmp.extend(train_pos_col_interval[i])
                    tmp.extend(train_neg_col_interval[i])
                    bin = preprocess.get_bin_value(tmp, part)
                    segmentation_process_sequence.append(["discrete", bin])
                    # 记录 bin
                    process_info_result.append("interval-{0} discrete_value : ".format(i) + json.dumps({"bin": "|".join([str(one) for one in bin])}))

                elif strategy.startswith("discrete_bin"):
                    boundary_list = strategy.split(":")[1].split("-")
                    dimension.append(len(boundary_list) + 1)
                    bin = [float(one) for one in boundary_list]
                    segmentation_process_sequence.append(["discrete", bin])
                    # 记录 bin
                    process_info_result.append("interval-{0} discrete_bin : ".format(i) + json.dumps({"bin": "|".join([str(one) for one in bin])}))

                else:
                    print "offline_exp.py - do_process() : strategy-{0} error.".format(strategy)
                    exit()

            return preprocess.segmentation_process(segmentation_process_sequence, dimension, interval, train_pos_col),\
                   preprocess.segmentation_process(segmentation_process_sequence, dimension, interval, train_neg_col),\
                   preprocess.segmentation_process(segmentation_process_sequence, dimension, interval, test_pos_col),\
                   preprocess.segmentation_process(segmentation_process_sequence, dimension, interval, test_neg_col),\
                   process_info_result

        else:
            print "offline_exp.py - do_process() : strategy-{0} error.".format(strategy)
            exit()




    def run(self, train_pos_processed, train_neg_processed, test_pos_processed, test_neg_processed):
        if not self.feature_process_dict["General"].has_key('select_features'):
            print 'model_LR has no option select_features'
            exit()
        features_select = self.feature_process_dict["General"]['select_features'].strip().split(',')

        feature_matrix_train_pos = [[] for i in range(len(self.X_train_pos))]
        feature_matrix_train_neg = [[] for i in range(len(self.X_train_neg))]
        feature_matrix_test_pos = [[] for i in range(len(self.X_test_pos))]
        feature_matrix_test_neg = [[] for i in range(len(self.X_test_neg))]

        rt_process_info_list = []
        rt_processed_length = []
        for feature in features_select:

            idx = features_select.index(feature)
            #if idx == 0 or idx == 1:
            #    continue

            process_info = self.feature_process_dict[feature]
            if process_info.strip() == "":
                continue

            train_pos_col, train_neg_col, test_pos_col, test_neg_col, process_info_result = self.do_process(idx, feature, process_info)
            rt_process_info_list.append((idx, feature, process_info))
            rt_processed_length.append(len(train_pos_col[0]))

            for i in range(len(train_pos_col)):
                feature_matrix_train_pos[i].extend(train_pos_col[i])
            for i in range(len(train_neg_col)):
                feature_matrix_train_neg[i].extend(train_neg_col[i])
            for i in range(len(test_pos_col)):
                feature_matrix_test_pos[i].extend(test_pos_col[i])
            for i in range(len(test_neg_col)):
                feature_matrix_test_neg[i].extend(test_neg_col[i])

            record_process_info(self.result_dir, idx, feature, process_info_result)
            #break

        # 将处理完的特征写入文件
        record_feature_matrix(feature_matrix_train_pos, train_pos_processed)
        record_feature_matrix(feature_matrix_train_neg, train_neg_processed)
        record_feature_matrix(feature_matrix_test_pos, test_pos_processed)
        record_feature_matrix(feature_matrix_test_neg, test_neg_processed)

        return rt_process_info_list, rt_processed_length



#def run(conf_dict):
def run(Base_Dir_40p208, date, cid, bid, conf_dict, train_list, test_list):

    train_pos = Base_Dir_40p208 + "/../data/lr/feature_set/{0}/{1}/{2}/train_pos".format(cid, bid, date)
    train_neg = Base_Dir_40p208 + "/../data/lr/feature_set/{0}/{1}/{2}/train_neg".format(cid, bid, date)
    test_pos = Base_Dir_40p208 + "/../data/lr/feature_set/{0}/{1}/{2}/train_pos".format(cid, bid, date)
    test_neg = Base_Dir_40p208 + "/../data/lr/feature_set/{0}/{1}/{2}/train_neg".format(cid, bid, date)

    train_pos_processed = Base_Dir_40p208 + "/../data/lr/result/{0}/{1}/{2}/train_pos_processed".format(cid, bid, date)
    train_neg_processed = Base_Dir_40p208 + "/../data/lr/result/{0}/{1}/{2}/train_neg_processed".format(cid, bid, date)
    test_pos_processed = Base_Dir_40p208 + "/../data/lr/result/{0}/{1}/{2}/test_pos_processed".format(cid, bid, date)
    test_neg_processed = Base_Dir_40p208 + "/../data/lr/result/{0}/{1}/{2}/test_neg_processed".format(cid, bid, date)

    result_dir = Base_Dir_40p208 + "/../data/lr/result/{0}/{1}/{2}/".format(cid, bid, date)
    result_file = Base_Dir_40p208 + "/../data/lr/result/{0}/{1}/{2}/result".format(cid, bid, date)
    predict_file = Base_Dir_40p208 + "/../data/lr/result/{0}/{1}/{2}/predict".format(cid, bid,date)


    do_mkdir(Base_Dir_40p208 + "/../data/lr/feature_set/{0}".format(cid))
    do_mkdir(Base_Dir_40p208 + "/../data/lr/feature_set/{0}/{1}".format(cid, bid))
    do_mkdir(Base_Dir_40p208 + "/../data/lr/feature_set/{0}/{1}/{2}".format(cid, bid, date))
    do_mkdir(Base_Dir_40p208 + "/../data/lr/result/{0}".format(cid))
    do_mkdir(Base_Dir_40p208 + "/../data/lr/result/{0}/{1}".format(cid, bid))
    do_mkdir(Base_Dir_40p208 + "/../data/lr/result/{0}/{1}/{2}".format(cid, bid, date))

    # 本地开发
    '''
    train_pos = "./data/train_pos_dl"
    train_neg = "./data/train_neg_dl"
    test_pos = "./data/test_pos_dl"
    test_neg = "./data/test_neg_dl"
    train_pos_processed = "./data2/train_pos_processed"
    train_neg_processed = "./data2/train_neg_processed"
    test_pos_processed = "./data2/test_pos_processed"
    test_neg_processed = "./data2/test_neg_processed"
    result_dir = "./result/"
    result_file = "./result/result_file"
    predict_file = "./result/predict_file"
    '''

    process = PreProcess(train_pos, train_neg, test_pos, test_neg, result_dir, conf_dict)
    process_info_list, processed_length = process.run(train_pos_processed, train_neg_processed, test_pos_processed, test_neg_processed)

    coef, auc = lr.run(train_pos_processed, train_neg_processed, test_pos_processed, test_neg_processed, predict_file)

    writer = open(result_file, "w")
    writer.write("coef:\n")
    for i in range(len(processed_length)):
        writer.write("-------------------- " + process_info_list[i][1] + " --------------------\n")
        writer.write("--- " + process_info_list[i][2] + " ---\n")
        for weight in coef[0][sum(processed_length[:i]) : sum(processed_length[:i+1])]:
            writer.write("{0}\n".format(weight))
    writer.write("\nauc:\n")
    writer.write("{0}\n".format(auc))
    writer.close()

    # 将实验结果存入mysql
    save_model_lr.run(cid, bid, train_list, test_list, process_info_list, result_dir, result_file) 



if __name__ == "__main__":
    '''
    'normalize'

    'discrete_freq:13'
    'discrete_value:13'
    'discrete_bin:1000-5000-10000-20000'

    'discrete_special_freq:0-1-2 13'
    'discrete_special_value:0-1-2 13'
    'discrete_special_bin:0-1-2 1000-5000-10000-20000'

    'segmentation_bin:1000-5000-10000-20000;normalize;normalize;normalize;normalize;normalize'
    'segmentation_freq:5;normalize;normalize;normalize;normalize;normalize'
    'segmentation_value:5;normalize;normalize;normalize;normalize;normalize'
    '''
    conf_dict = {
                 'click_rate_15day': 'discrete_freq:13',
                 'click_rate_1day': 'discrete_freq:13',

                # 测试用例
                 'feedback_15day': 'segmentation_freq:3;normalize;normalize;discrete_bin:1000-5000-10000-20000',
                 'realresponse_15day': 'segmentation_freq:1;discrete_bin:1000-5000-10000-20000',
                 'visit_1day': 'discrete_special_bin:0-1-2 1000-5000-10000-20000',

                 'select_features': 'click_rate_15day,click_rate_1day,feedback_15day,realresponse_15day,visit_1day',
                 'sample_proportion': '1:7',
                 'test_days': '1',
                 'train_days': '3',
                  'sample': 'True'
                  }

    Base_Dir_40p208 = "/opt/bre/rec/feature_project/script"
    date='2015-11-01'
    cid = "Czgc_pc"
    bid = "949722CF_12F7_523A_EE21_E3D591B7E755"
    train_list = ['2015-10-29', '2015-10-28', '2015-10-27', '2015-10-26', '2015-10-25']
    test_list = ['2015-11-01', '2015-10-31', '2015-10-30']

    run(Base_Dir_40p208, date,cid, bid, conf_dict, train_list, test_list)
    #run(conf_dict)
