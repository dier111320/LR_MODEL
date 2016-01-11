# coding=utf-8

import get_conf
import sys

def run_model_LR(Base_Dir_40p208, Base_Dir_share, cid, bid, date):
    sys.path.append(Base_Dir_40p208 + '/model_LR')
    import prepare_data
    import offline_exp

    conf = get_conf.run(Base_Dir_40p208, date, cid, bid, "1")
    if conf.has_key("model_LR"):
        conf = conf["model_LR"]
    else:
        print "run_model.py - run_model_LR() : cid-{0}, bid-{1} has no model_LR section.".format(cid, bid)
        exit()

    conf_format = {}
    conf_format["General"] = {}
    for k in conf:
        if k == "select_features":
            conf_format["General"][k] = conf[k]
            continue
        if k == "sample_proportion":
            conf_format["General"][k] = conf[k]
            continue
        if k == "sample":
            conf_format["General"][k] = conf[k]
            continue
        if k == "train_days":
            conf_format["General"][k] = conf[k]
            continue
        if k == "test_days":
            conf_format["General"][k] = conf[k]
            continue
        conf_format[k] = conf[k]

    train_list, test_list = prepare_data.run(Base_Dir_40p208, Base_Dir_share, date, cid, bid, conf_format)
    print train_list
    print test_list

    offline_exp.run(Base_Dir_40p208, date, cid, bid, conf_format, train_list, test_list)

if __name__ == "__main__":
#    Base_Dir_40p208=sys.argv[1]
#    run_date=sys.argv[3]

    #Base_Dir_40p208 = "/opt/bre/rec/feature_project/script"
    #Base_Dir_share='/opt/share/feature_project/script'
    #run_date = "2015-11-01"

    # Base_Dir_40p208 = sys.argv[1]
    # Base_Dir_share = sys.argv[2]
    # cid = sys.argv[3]
    # bid = sys.argv[4]
    # run_date = sys.argv[5]
    Base_Dir_40p208 = '/opt/bre/rec/feature_project/script'
    Base_Dir_share='/opt/share/feature_project/script'
    date='2015-11-26'
    cid = 'Czgc_pc'
    bid = '949722CF_12F7_523A_EE21_E3D591B7E755'
    conf_dict={'visit_1day': 'discrete,freq,17', 'select_features': 'click_rate_15day,click_rate_1day,feedback_15day,realresponse_15day,visit_1day', 'sample_proportion': '1:7', 'test_days': '1', 'feedback_15day': 'discrete,freq,10', 'train_days': '1', 'realresponse_15day': 'discrete,freq,13', 'sample': 'True', 'click_rate_1day': 'normalize', 'click_rate_15day': 'normalize'}

    run_model_LR(Base_Dir_40p208, Base_Dir_share, cid, bid, run_date)


