# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import SMOTE


def format_X_y(pos_file, neg_file,flag):
    #flag = 'train' or 'test'
    #print pos_file, neg_file
    X, y = [], []
    for line in open(pos_file):
        try:
            x = [float(item) for item in line.strip().split()]
            X.append(x)
            y.append(1)
        except:
            print line


    if flag == 'train':
    #using SMOTE to over sample the positive feature vectors. 
        X=SMOTE.smote(X,15,3)
        length1 = len(X)-len(y)
        y+=[1]*length1

 
    #down sample part 
    tempX,tempy = [],[]
    for line in open(neg_file):
        try:
            x = [float(item) for item in line.strip().split()]
            tempX.append(x)
            tempy.append(0)
        except:
            print line
    '''if flag == 'train':
        tempX = SMOTE.downsample(tempX,0.5)
        tempy = tempy[:len(tempX)]'''
    #print float(len(tempy))/float(len(y))
    X+=tempX
    y+=tempy
    
    return X, y

def train(X, y):
    model = LogisticRegression(penalty='l1')
    #model = RandomForestClassifier()
    model.fit(X, y)
    return model

def predict(lr, X):
    proba = lr.predict_proba(X)
    return [item[1] for item in proba]

def auc(lr, test_pos_file, test_neg_file, predict_file):
    X, y = format_X_y(test_pos_file, test_neg_file,'test')
    writer = open(predict_file, 'w')
    _y = predict(lr, X)
    #print "bias: "
    #print lr.intercept_

    for res in _y:
        writer.write(str(res) + '\n')
    writer.close()
    fpr, tpr, thresholds = metrics.roc_curve(np.array(y), np.array(_y))
    return metrics.auc(fpr, tpr)

def run(pos_file, neg_file, test_pos_file, test_neg_file, predict_file):
    X, y = format_X_y(pos_file, neg_file,'train')
    lr = train(X, y)
    #print lr.densify().coef_
    auc_value = auc(lr, test_pos_file, test_neg_file, predict_file)

    return lr.densify().coef_, auc_value


if __name__ == "__main__":
    print run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
