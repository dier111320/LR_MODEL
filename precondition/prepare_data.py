# coding=utf-8

import os
import sys
import datetime
import collections
import random

def get_date(num):
    '''
    获取距离今日num天前的日期
    e.g. 今天2015-10-20，get_date(1)返回2015-10-19
    '''
    date_value = (datetime.datetime.now()-datetime.timedelta(minutes=num*1440)).strftime("%Y-%m-%d").__str__()
    return date_value

def get_date2(end_str, num):
    '''
    获取end_str之前num天的日期，end_str是字符串类型
    '''
    try:
        year = int(end_str.split("-")[0])
        month = int(end_str.split("-")[1])
        day = int(end_str.split("-")[2])
    except:
        print "prepare_data.py - get_date2() : error input param end_str : " + end_str
        exit()
    end_date = datetime.date(year, month, day)
    return (end_date - datetime.timedelta(num)).__str__()

def get_common_fields(Base_Dir_208, date, cid, bid, type):
    sys.path.append(Base_Dir_208 + '/run')
    import get_conf as gc

    try:
        common_fields = gc.run(Base_Dir_208, date, cid, bid, type)["General"]["common_fields"]
        fields_list = []
        for field in common_fields.split(","):
            field = field.strip()
            if field != "":
                fields_list.append(field)
        return fields_list
    except:
        print "prepare_data.py - get_common_fields() : parse feature_coherent.conf file error"
        exit()


def get_feature_columns(head_features, feature_file, tail_features, conf):

    if not os.path.exists(feature_file):
        print "prepare_data.py - get_feature_columns() : feature file does not exist : {0}".format(feature_file)
        exit()

    features_all = []
    for feat in head_features:
        features_all.append(feat)

    with open(feature_file) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            chunks = line.split(",")
            for chunk in chunks:
                chunk = chunk.strip()
                if len(chunk) == 0:
                    continue
                features_all.append(chunk)
    for feat in tail_features:
        features_all.append(feat)

    features_select = []
    if not conf["General"].has_key('select_features'):
        print 'prepare_data.py - get_feature_columns() : conf file has no option select_features'
        exit()
    features_select = conf["General"]['select_features'].strip().split(',')

    features_name = []
    features_index = []
    for feat in features_select:
        if feat not in features_all:
            print "prepare_data.py - get_feature_columns() : selected feature-{0} not in all features.".format(feat)
            print "all features : ", " , ".join(features_all)
            exit()
        features_name.append(feat)
        features_index.append(features_all.index(feat))

    return features_name, features_index


def get_train_test_date(date, train_num, test_num):
    train_list = []
    test_list = []

    for i in range(0, int(test_num)):
        test_list.append(get_date2(date, i))

    train_start_date = get_date2(date, int(test_num))
    for i in range(0, int(train_num)):
        train_list.append(get_date2(train_start_date, i))

    return train_list, test_list


def feature_select(Base_Dir_208, Base_Dir_share, deal_date, cid, bid, features_index, label_index):
    do_mkdir(Base_Dir_208 + "/../data/lr/feature_oneday/{0}/{1}/{2}".format(cid, bid, deal_date))
#       此处使用处理过null值的文件，等以后datareplace转移一个路径，再改一下位置
    integration_file = Base_Dir_share + "/../data/feature_integration/{0}/{1}/{2}.integration".format(deal_date, cid, bid)
    feature_file_all = Base_Dir_208 + "/../data/lr/feature_oneday/{0}/{1}/{2}/feature_selection.all".format(cid, bid, deal_date)
    feature_file_label = Base_Dir_208 + "/../data/lr/feature_oneday/{0}/{1}/{2}/feature_selection.label".format(cid, bid, deal_date)
    feature_file_pos = Base_Dir_208 + "/../data/lr/feature_oneday/{0}/{1}/{2}/feature_selection.pos".format(cid, bid, deal_date)
    feature_file_neg = Base_Dir_208 + "/../data/lr/feature_oneday/{0}/{1}/{2}/feature_selection.neg".format(cid, bid, deal_date)

    features_index_str = [str(one+1) for one in features_index]
    try:
        cmd = 'awk -F"\\t"' + " '{print $" + '"\\t"$'.join(features_index_str) + "}' " + integration_file + " > " + feature_file_all
        os.system(cmd)
    except:
        print "prepare_data.py - feature_select() : cmd awk error :"
        print cmd
        exit()
    try:
        cmd = '''awk -F"\\t" '{print $''' + str(label_index+1) + "}' " + integration_file + " > " + feature_file_label
        os.system(cmd)
    except:
        print "prepare_data.py - feature_select() : cmd awk error :"
        print cmd
        exit()

def gen_pos_neg(Base_Dir_208,Base_Dir_share,deal_date,cid,bid,rate):
    feature_file_all = Base_Dir_208 + "/../data/lr/feature_oneday/{0}/{1}/{2}/feature_selection.all".format(cid, bid, deal_date)
    feature_file_label = Base_Dir_208 + "/../data/lr/feature_oneday/{0}/{1}/{2}/feature_selection.label".format(cid, bid, deal_date)
    feature_file_pos = Base_Dir_208 + "/../data/lr/feature_oneday/{0}/{1}/{2}/feature_selection.pos".format(cid, bid, deal_date)
    feature_file_neg = Base_Dir_208 + "/../data/lr/feature_oneday/{0}/{1}/{2}/feature_selection.neg".format(cid, bid, deal_date)

    pos_list = []
    pos_dict = {}
    neg_list = []
    neg_dict = {}
    count_pos = 0
    count_neg = 0
    with open(feature_file_all) as f1, open(feature_file_label) as f2:
        for line1 in f1:
            line1 = line1.strip()
            line2 = f2.readline().strip()
            if line2 == "1":
                pos_list.append(count_pos)
                pos_dict[count_pos] = line1
                count_pos += 1
            elif line2 == "0":
                neg_list.append(count_neg)
                neg_dict[count_neg] = line1
                count_neg += 1
            else:
                print "prepare_data.py - gen_pos_neg() : error label {0} in file {1}".format(line2, feature_file_label)

    if rate != 0:
        random.shuffle(neg_list)
        num = int(len(pos_list)*rate)
        neg_list = neg_list[:num]

    data_list = []
    for count_pos in pos_list:
        data_list.append(pos_dict[count_pos])
    writer = open(feature_file_pos, "w")
    for one in data_list:
        writer.write(one + "\n")
    writer.close()

    data_list = []
    for count_neg in neg_list:
        data_list.append(neg_dict[count_neg])
    writer = open(feature_file_neg, "w")
    for one in data_list:
        writer.write(one + "\n")
    writer.close()

def get_sample_rate(conf):
    rate = 0
    if not conf["General"].has_key('sample'):
        return rate
    if conf["General"]['sample']=='False':
        return rate
    if not conf["General"].has_key('sample_proportion'):
        return rate
    else:
        try:
            sample_proportion = conf["General"]['sample_proportion'].strip().split(':')
            rate = int(sample_proportion[1])/float(sample_proportion[0])
            return rate
        except:
            return 0


def do_mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    else:
        pass

def do_remove(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        pass

def get_combine(Base_Dir_208,date,cid,bid,train_list,test_list):
    # 合并多天的特征子集文件（正负例），形成训练、测试集
    train_pos = Base_Dir_208 + "/../data/lr/feature_set/{0}/{1}/{2}/train_pos".format(cid, bid, date)
    train_neg = Base_Dir_208 + "/../data/lr/feature_set/{0}/{1}/{2}/train_neg".format(cid, bid, date)
    test_pos = Base_Dir_208 + "/../data/lr/feature_set/{0}/{1}/{2}/test_pos".format(cid, bid, date)
    test_neg = Base_Dir_208 + "/../data/lr/feature_set/{0}/{1}/{2}/test_neg".format(cid, bid, date)
    do_mkdir(Base_Dir_208 + "/../data/lr/feature_set/{0}/".format(cid))
    do_mkdir(Base_Dir_208 + "/../data/lr/feature_set/{0}/{1}".format(cid, bid))
    do_mkdir(Base_Dir_208 + "/../data/lr/feature_set/{0}/{1}/{2}".format(cid, bid, date))
    do_remove(train_pos)
    do_remove(train_neg)
    do_remove(test_pos)
    do_remove(test_neg)
    for deal_date in train_list:
        pos = Base_Dir_208 + "/../data/lr/feature_oneday/{0}/{1}/{2}/feature_selection.pos".format(cid, bid, deal_date)
        neg = Base_Dir_208 + "/../data/lr/feature_oneday/{0}/{1}/{2}/feature_selection.neg".format(cid, bid, deal_date)
        cmd = "cat {0} >> {1}".format(pos, train_pos)
        os.system(cmd)
        cmd = "cat {0} >> {1}".format(neg, train_neg)
        os.system(cmd)
    for deal_date in test_list:
        pos = Base_Dir_208 + "/../data/lr/feature_oneday/{0}/{1}/{2}/feature_selection.pos".format(cid, bid, deal_date)
        neg = Base_Dir_208 + "/../data/lr/feature_oneday/{0}/{1}/{2}/feature_selection.neg".format(cid, bid, deal_date)
        cmd = "cat {0} >> {1}".format(pos, test_pos)
        os.system(cmd)
        cmd = "cat {0} >> {1}".format(neg, test_neg)
        os.system(cmd)


def run(Base_Dir_208, Base_Dir_share, date, cid, bid, conf):
    try:
        train_days = conf['General']['train_days']
        test_days = conf['General']['test_days']
    except:
        train_days = 1
        test_days = 1
        print 'prepare_data.py - run() : conf file has no option train_days,test_days'
    try:
        train_days = int(train_days)
        test_days = int(test_days)
    except:
        print 'prepare_data.py - run() : conf file option error : train_days-{0} or test_days-{1}'.format(train_days, test_days)


    # 获取训练、测试日期
    train_list, test_list = get_train_test_date(date, train_days, test_days)
    
    # 获取每天的特征子集文件（全部样本、正、负样本，共3份）
    do_mkdir(Base_Dir_208 + "/../data/lr/feature_oneday/{0}/".format(cid))
    do_mkdir(Base_Dir_208 + "/../data/lr/feature_oneday/{0}/{1}".format(cid, bid))

    for deal_date in train_list + test_list:
        # "个性"特征 
        feature_file = Base_Dir_208 + "/../conf/default/{0}/{1}/{2}.feature".format(deal_date, cid, bid)
        # "共性"特征 
        co_features = get_common_fields(Base_Dir_208, deal_date, cid, bid, "3")
        # 获取特征子集（select），index:在全部特征中的索引 
        tail_features = ["date"]
        features_name, features_index = get_feature_columns(co_features, feature_file, tail_features, conf)

        label_index = co_features.index("label")
        rate = get_sample_rate(conf)

        print "Deal :", deal_date
        feature_select(Base_Dir_208, Base_Dir_share, deal_date, cid, bid, features_index, label_index)
        gen_pos_neg(Base_Dir_208, Base_Dir_share, deal_date, cid, bid, rate)

    get_combine(Base_Dir_208, date, cid, bid, train_list, test_list)

    return train_list, test_list




if __name__ == "__main__":

    # 临时使用
    #train_list, test_list = get_train_test_date('2015-11-01', 5, 3)
    #print train_list
    #print test_list
    #exit()

    Base_Dir_208 = "/opt/bre/rec/feature_project/script"
    date='2015-11-26'
    cid = "Czgc_pc"
    bid = "949722CF_12F7_523A_EE21_E3D591B7E755"
    conf_dict={'visit_1day': 'discrete,freq,17', 'select_features': 'click_rate_15day,click_rate_1day,feedback_15day,realresponse_15day,visit_1day', 'sample_proportion': '1:7', 'test_days': '1', 'feedback_15day': 'discrete,freq,10', 'train_days': '1', 'realresponse_15day': 'discrete,freq,13', 'sample': 'True', 'click_rate_1day': 'normalize', 'click_rate_15day': 'normalize'}

    run(Base_Dir_208,Base_Dir_share,date, cid, bid, conf_dict)

