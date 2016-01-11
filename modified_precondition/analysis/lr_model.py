#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys,re,math,codecs,numpy
# from random import Random
reload(sys)
sys.setdefaultencoding('utf8')
# sys.path.append("")
import pandas as pd
import os
import string
import datetime
import operator
import random
from random import random
from datetime import datetime
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
#from treeinterpreter import treeinterpreter as ti
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
pd.set_option('expand_frame_repr', False)

import getopt,sys,subprocess,time
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from sklearn import preprocessing

global feature_analysis
feature_analysis=True
stats_startdate='2015-12-29'
deal_date='2015-12-31'
deal_table='bosszp.bosszp_recommend_traindata'
config={
    'hive':'/home/hive/hive/bin/hive',
    'dbhost':'192.168.254.105',
    'dbport':3306,
    'dbname':'bidb',
    'dbuser':'bdp',
    'dbpwd':'kanzhun,.bdp',
}



def Classification(df_detail,features,featurey,featue_selection):

    df_X=df_detail[features]
    # print df_X.isnull().values.any()
    X=numpy.array(df_X)

    # from sklearn.preprocessing import StandardScaler
        # scaler = StandardScaler()
        # X = scaler.fit_transform(X)

    Y=list(df_detail[featurey])


    # print 'lenY',len(Y)


    # ############################################
    # #  classification
    # ############################################
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.3, random_state=0)
    # print 'X_train[1]:',X_train[1]


    #####LR######
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    expected = y_test
    predicted = lr.predict(X_test)
    answer=lr.predict_proba(X_test)
    if featue_selection==True:
        prob_auc=pd.DataFrame({'feature':features,
                               'auc':roc_auc_score(numpy.array( map(int, y_test)), answer[:,1])})
        return prob_auc
    else:
        print '=====LogisticRegression======'
        print '1/0 in train:%d/%d\t1/0 in test:%d/%d'%(y_train.count('1'),y_train.count('0'),y_test.count('1'),y_test.count('0'))
        print 'N_train:N_test= %d:%d'%(len(y_train),len(y_test))
        print(metrics.classification_report(expected, predicted))
        print(metrics.confusion_matrix(expected, predicted))
        print 'lr.score=',lr.score(X_test,y_test)
        print 'lr.auc_score=',roc_auc_score(numpy.array( map(int, y_test)), answer[:,1])

    # print '****lr.coef_****'
    # print lr.coef_


        lr_coef=pd.DataFrame(lr.coef_)
        lr_coef.to_csv('lr_coef_new.txt',sep='\t' ,index=False, header=False)
        lr_intercept=pd.DataFrame(lr.intercept_)
        lr_intercept.to_csv('lr_intercept_new.txt',sep='\t' ,index=False, header=False)

        test_pair=pd.concat([pd.DataFrame(y_test),pd.DataFrame(answer),pd.DataFrame(X_test)],axis=1)
        test_pair.to_csv('test_pair_new.txt',sep='\t' ,index=False, header=False)


        feature_imp=pd.DataFrame(lr.coef_[0]
        )
        # print feature_imp
        feature_imp.to_csv('feature_imp.txt',sep='\t', mode='a',index=False, header=False)
def hive(hiveSql):
    r=[]
    cmd = [config['hive'],'-S','-e',hiveSql]
    process = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=None)
    while True:
        line = process.stdout.readline()
        if line:
            r.append(line.strip())
        else:
            break
    process.kill()
    return r
def Hive2DF(hql):
    print 'Hive sql:',hql
    data = hive(hql)
    # print 'Hive result:',data

    clean_data = [[item.split('\t') for item in row.split(',')] for row in data]
    clean_data=pd.DataFrame(clean_data)
    clean_data.columns=['x']
    clean_data=pd.DataFrame(clean_data['x'].tolist())
    clean_data=clean_data.convert_objects(convert_numeric=True)
    return clean_data


def GeekActInfo():
    #
    # hql='select /*+mapjoin(a)*/a.geek_id,b.date_time as time,b.action,b.ds as date,unix_timestamp(b.date_time,"yyyy-MM-dd HH:mm:ss") unixtime, a.deal_date,a.deal_time,unix_timestamp(a.deal_time,"yyyy-MM-dd HH:mm:ss")deal_unixtime from %s a  join bosszp.bg_action  b  on a.geek_id=b.uid and b.bg=0 and b.ds>= "%s"  where  a.ds = "%s" and a.deal_time>=b.date_time and a.flag in (1 ) GROUP BY a.geek_id,b.date_time,b.action,b.ds,a.deal_date,a.deal_time limit 3000;  '%(deal_table,stats_startdate,deal_date)
    # hql='select /*+mapjoin(a)*/a.geek_id,b.date_time as time,b.action,b.ds as date,unix_timestamp(b.date_time,"yyyy-MM-dd HH:mm:ss") unixtime, a.deal_date,max(a.deal_time),unix_timestamp(max(a.deal_time),"yyyy-MM-dd HH:mm:ss")deal_unixtime from %s a  join bosszp.bg_action  b  on a.geek_id=b.uid and b.bg=0 and b.ds>= "%s"  where  a.ds = "%s" and a.deal_time>=b.date_time and a.flag in (1 ) GROUP BY a.geek_id,b.date_time,b.action,b.ds,a.deal_date limit 3000;  '%(deal_table,stats_startdate,deal_date)
    hql='select /*+mapjoin(a)*/a.geek_id,b.date_time as time,b.action,b.ds as date,unix_timestamp(b.date_time,"yyyy-MM-dd HH:mm:ss") unixtime, a.deal_date,concat(a.deal_date," 23:59:59"),unix_timestamp(concat(a.deal_date," 23:59:59"),"yyyy-MM-dd HH:mm:ss")deal_unixtime from %s a  join bosszp.bg_action  b  on a.geek_id=b.uid and b.bg=0 and b.ds>= "%s"  where  a.ds = "%s" and a.deal_time>=b.date_time and a.flag in (0,1 )  GROUP BY a.geek_id,b.date_time,b.action,b.ds,a.deal_date ;  '%(deal_table,stats_startdate,deal_date)

    geek_act=Hive2DF(hql)
    geek_act.columns=['geek_id','time','action','date','unixtime','deal_date','max_deal_time','max_deal_unixtime']
    geek_act.to_csv('geek_act.txt',header=True)
    hql='select distinct   geek_id, deal_date, deal_time, concat( deal_date," 23:59:59")max_deal_time from %s   where ds ="%s" and  flag in (0,1)    '%(deal_table,deal_date)
    deal_list=Hive2DF(hql)
    deal_list.columns=['geek_id','deal_date','deal_time','max_deal_time']
    deal_list.to_csv('deal_list.txt',header=True)


    print 'get data'
    # print geek_act.dtypes
    # print geek_act

    return geek_act,deal_list

def GetBehaviorStatistics(action_list,expire_intl):
    geek_act,deal_list=GeekActInfo()
    geek_act=pd.read_table("geek_act.txt", sep=",", na_values="NULL")
    geek=geek_act.sort(['deal_date','geek_id','unixtime','date'], ascending=[True,True,True,True])
    geek['index']=list(geek.index)
    first_row = geek.geek_id != geek.geek_id.shift(1)
    first_row=pd.DataFrame(first_row)
    first_row.columns=['first_row']
    geek=pd.concat([geek,first_row], axis=1)
    # print 'diff'
    geek_act_diff=pd.DataFrame(geek.unixtime.diff()).reset_index()
    geek=pd.merge(geek,geek_act_diff,on='index',how='left')
    # print geek
    geek.columns=['ind','geek_id','time','action','date','unixtime','deal_date','max_deal_time','deal_unixtime','index','first_row','intl']
    geek['index']=geek.index

    geek=geek.fillna(0)

    geek=pd.merge(geek,deal_list,on=['max_deal_time','geek_id','deal_date'],how='left')
    geek=geek.loc[(geek['time']<=geek['deal_time']),] #3d

    out=geek.loc[geek['intl']>expire_intl,['geek_id','deal_time']] #3d
    out['index']= list(out.index)
    latest_out=out.groupby(['geek_id','deal_time'])['index'].idxmax().reset_index()
    # print 'latest_out'
    # print latest_out
    if len(latest_out)>0:
        merge=pd.merge(geek,latest_out,on=['geek_id','deal_time'],how='left')
        # print merge
        geek=geek.loc[geek.index>=merge['index_y'],]
    YN_action=geek['action'].isin(action_list) #鍙绠楅渶瑕佺殑缁熻閲�
    if YN_action.sum()>0:
        geek=geek[geek['action'].isin(action_list)]
        print 'actions:',geek.head()
        # print '##########group sum##########'
        pivot= pd.pivot_table(geek,values='intl',index=['deal_date','deal_time','geek_id'],columns='action',aggfunc=len)
        action_count=pivot.reset_index().fillna(0)
        # print action_count
    action_count.to_csv('behave_stats.txt',sep='\t',index=False,header=False)
    hql='load data local inpath "behave_stats.txt" overwrite into table wqy.action_statistics  ; '
    hive(hql)
    # print action_count

def GetAllFeaures():
    hql='select distinct b.flag,b.deal_date,b.pk_class,b.sys ,b.boss_id,b.geek_id,b.job_id,b.page,b.rank,b.list_time,b.deal_type,b.deal_time, if( a.app_active is null,0,a.app_active) as app_active, if( a.chat is null,0,a.chat) as chat, if( a.chat_read is null,0,a.chat_read) as  chat_read, if( a.detail_boss is null,0,a.detail_boss) as  detail_boss, if( a.list_boss is null,0,a.list_boss) as  list_boss, if( a.list_notify is null,0,a.list_notify) as  list_notify, if( a.msg_return is null,0,a.msg_return) as  msg_return, (unix_timestamp(cast(b.deal_date  as string),"yyyy-MM-dd")-unix_timestamp(cast(bs.boss_date8 as string),"yyyyMMdd"))/(24*3600) boss_days, (unix_timestamp(cast(b.deal_date  as string),"yyyy-MM-dd")-unix_timestamp(cast(bs.date8  as string),"yyyyMMdd"))/(24*3600) job_days, (unix_timestamp(cast(b.deal_date  as string),"yyyy-MM-dd")-unix_timestamp(cast(g.date8  as string),"yyyyMMdd"))/(24*3600) geek_days from %s   b  join bosszp.zp_geek g on b.geek_id=g.geek_id join bosszp.zp_job bs on b.boss_id=bs.boss_id and b.job_id=bs.id left join wqy.action_statistics a  on a.deal_date=b.deal_date and a.deal_time=b.deal_time and a.geek_id=b.geek_id  where b.ds="%s" and b.flag in (0,1 ); '%(deal_table,deal_date)
    fea=Hive2DF(hql)
    if len(fea)==0:
        print 'Error: no match'
    fea.columns=['flag','deal_date','pk_class','sys','boss_id','geek_id','job_id','page','rank','list_time','deal_type','deal_time','app_active','chat','chat_read','detail_boss','list_boss','list_notify','msg_return','boss_days','job_days','geek_days']
    return fea
def CtrPlot(resultDir,data,ctrbin,feature):
    data=pd.DataFrame(data)
    row, col = data.shape
    if col == 2 and type(ctrbin) == int:
        data=data.astype(float)
        data.columns=['feature','label']
        data=data.sort_values(by='feature')
        featuredata=data['feature']
        label=data['label']
        hist, edges=numpy.histogram(range(row),bins=ctrbin)
        edges1=[int(math.ceil(i)) for i in edges]
        ctrx=list(featuredata.iloc[edges1])
        ctrx1=[str(round((ctrx[i]+ctrx[i+1])/2,2)) for i in range(len(ctrx)-1)]
        ctr=[sum(label[edges1[i]:edges1[i+1]])/numpy.float(hist[i]) for i in range(len(edges1)-1)]
        plt.plot(ctr)
        plt.annotate('sample size: '+str(hist[0]),xy=(ctrbin-6, max(ctr)-0.01))
        plt.xticks(range(ctrbin),ctrx1,rotation=25)
        plt.ylabel('CTR')
        plt.xlabel('Feature: '+feature)
        plt.title('Equal Frenquency Distribution Map')
        plt.savefig(resultDir+'ctr_'+feature+'.png')
        plt.cla()
        plt.clf()
def HistPlot(resultDir,data,bin,feature,logx,logy):
    featuredata=data[feature].astype(float)
    smin,smax=tuple(featuredata.describe(percentiles=[0.98])[['min','98%']])
    featuredata=featuredata[featuredata<=smax]
    featuredata.plot(kind='hist',bins=bin,logx=logx,logy=logy,title=feature,legend=False)
    plt.xlabel('Feature: '+feature)
    plt.ylabel('Number Of Interval')
    plt.title('Histogram')
    plt.savefig(resultDir+'hist_'+feature+'.png')
    plt.cla()
    plt.clf()

def RemoveOutlier(data,basic_features,type):
    if isinstance(type,float):
        return data[data[basic_features].apply(lambda x: x<x.quantile(type)).all(axis=1)]
    else:
        return data[data[basic_features].apply(lambda x: numpy.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
def Discretize(data,feature,bins):
    group_names=[(feature+'_'+str(b))  for b in bins[1:len(bins)] ]
    print group_names
    dummy=pd.get_dummies(pd.cut(data[feature],bins,labels=group_names,include_lowest=True))
    data=pd.concat([data, dummy], axis=1)
    return data
def Binarize(data,feature):
    tmp=pd.get_dummies(data[feature], prefix=feature)
    return pd.concat([data, tmp], axis=1)
def Pivot(data,feature,featurey):
    pivot= pd.pivot_table(data.filter(regex=r'^('+feature+'_|'+featurey+').*', axis=1),index=[featurey],aggfunc=numpy.sum).reset_index()
    pivot= pivot.transpose()
    pivot.to_csv('pivot.txt',sep='\t',header=False,mode='a')
def Correlation(data,features):
    corr=numpy.corrcoef(data,y=None, rowvar=False)
    corrdf=pd.DataFrame(corr)
    corrdf.columns=features
    corrdf.index=features
    nrow=corrdf.shape[0]
    corrlist=[]
    for i in range(1,nrow):
        for j in range(i):
            corrlist.append([corrdf.index[i],corrdf.columns[j],round(corrdf.iloc[i,j],2)])
    corr_result=pd.DataFrame(corrlist)
    corr_result.columns=['feature_name1','feature_name2','corr_value']
    corr_result.to_csv('corr_result.txt',sep=' ')
    # print corr_result
def Replication(fea):
    chat=fea.loc[fea['deal_type']=='chat',]
    chat2 = chat.loc[numpy.repeat(chat.index.values,100)]
    print len(chat),len(chat2)
    addf=fea.loc[fea['deal_type']=='addf',]
    addf2 = addf.loc[numpy.repeat(addf.index.values,50)]
    print len(addf),len(addf2)
    fea=pd.concat([fea,addf2,chat2])
    return fea
if __name__ == "__main__":
    # print '--------------------GEEK STATISTICS--------------------'
    # action_list=['app-active','chat','chat-read','detail-boss','list-boss','list-notify','msg-return'] #缁熻閲忔秹鍙婄殑action
    # expire_intl=3*24*3600
    # GetBehaviorStatistics(action_list,expire_intl)
    # print '--------------------GEEK STATISTICS+GEEK-BOSS DEAL INFO+GEEK-BOSS REGISTRATION INFO--------------------'
    # fea=GetAllFeaures()
    # fea.to_csv('features_0104.txt',index=False)



    print '****************FEATURE ANALYSIS****************'


    # f=open('pivot.txt','w+')
    # f.truncate()

    filelist=[file for file in os.listdir("base_data") if file.startswith("features_")]
    df_list= [pd.read_table('base_data/'+file, sep=",", na_values="NULL") for file in filelist]
    fea0 = pd.concat(df_list)
    print fea0.head()


    fea0['list_time']= fea0['list_time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    fea0['list_hour']= fea0['list_time'].apply(lambda x: x.hour)
    fea0['list_weekday']= fea0['list_time'].apply(lambda x: x.weekday()+1)

    fea0=fea0.loc[(fea0['list_hour']>=5),]
    fea=fea0.filter(regex=r'^(?!.*?(date|time|id)).*$', axis=1)
    print fea.columns



    print '-------------OUTLIERS-------------'
    basic_features=['list_hour',
            'page','rank','boss_days','job_days','geek_days',
              'app_active','chat','chat_read','detail_boss','list_boss','list_notify','list_weekday','msg_return'
              ]
    fea=RemoveOutlier(fea,basic_features,'std') #a float like 0.98 or std
    print 'num sample:',len(fea)


    fea['flag']=numpy.where(fea['flag']==0,'1','0')
    featurey='flag'
    if feature_analysis==True:
        print '--------------REPLICATE SAMPLES---------'
        fea=Replication(fea)
        print 'final length:',len(fea)

        print '--------------------CTR PLOT-------------------'
        for feature in basic_features:
            HistPlot('plots_after_replication/',fea,15,feature,False,False)
            CtrPlot('plots_after_replication/',fea[[feature,'flag']],15,feature)
        print '--------------Correlation---------'
        # Correlation(fea,basic_features)
        # print '--------------AUC---------'
        # prob_auc_list=[Classification(fea[[feature,featurey]],[feature],featurey,True) for feature in basic_features]
        # auc=pd.concat(prob_auc_list)
        # auc=auc.sort_values(by='auc',ascending=[False])
        # auc.to_csv('single_auc.txt',sep='\t')
    # else:
    #
    #     print '****************FEATURE PROCESSING****************'
    #
    #
    #
    #
    #     print '-------------BINARIZE-------------'
    #     binarize_vars=['sys','pk_class']
    #     for var in binarize_vars:
    #         fea=Binarize(fea,var)
    #     print fea.columns
    #     print '-------------DISCRETE----------------'
    #
    #     discret_vars = {}
    #     with open("data/discretize_conf.txt") as f:
    #         for line in f:
    #            (key, val) = line.split()
    #            discret_vars[key] = map(float,val.split(','))
    #            if key=='list_hour':
    #                continue
    #            discret_vars[key].append(numpy.Inf)
    #     for key  in  discret_vars.keys():
    #         fea=Discretize(fea,key,discret_vars[key])
    #     print fea.columns
    #
    #
    #     print '-------------SUMMARIZE FEATURES----------------'
    #     features=pd.Series(fea.columns)
    #     correl_vars=['chat','chat_read','detail_boss','list_boss','rank']
    #     to_remove=binarize_vars+discret_vars.keys()+correl_vars
    #     to_remove.append(featurey)
    #     # ['sys','pk_class','deal_type','flag','boss_days','geek_days','job_days','list_hour','app_active','chat','chat_read','detail_boss','list_boss','list_notify','msg_return','rank']
    #     features=features[~features.isin(to_remove)]
    #     print 'removed',len(to_remove),'features:',to_remove
    #     print features
    #
    #     print '--------------REPLICATE SAMPLES---------'
    #     fea=Replication(fea)
    #     print 'final length:',len(fea)
    #
    #     print '-------------NORMALIZE----------------'
    #
    #     # fea['msg_return'] =preprocessing.scale(fea['msg_return'])
    #     # print fea['msg_return'].describe()
    #     min_max_scaler = preprocessing.MinMaxScaler()
    #     fea['msg_return'] = min_max_scaler.fit_transform(fea['msg_return'])
    #
    #     # fea['msg_return'] =preprocessing.normalize(numpy.array(fea['msg_return']).reshape(len(fea),1), copy=False)
    #     print fea['msg_return'].describe()
    #
    #     print '############BASIC DESCRIPTION############'
    #     final_features=features[~features.isin(['deal_type'])]
    #     print "features.describe(percentiles=[0.98])"
    #     print fea[final_features].describe(percentiles=[0.98])
    #     print "features.skew"
    #     print fea[final_features].apply(lambda x: x.skew())
    #     print "features.kurt"
    #     print fea[final_features].apply(lambda x: x.kurt())
    #
    #
    #     print '############FEATURE ANALYSIS after replication#################'
    #     Pivot(fea,'boss_days',featurey)
    #     Pivot(fea,'geek_days',featurey)
    #     Pivot(fea,'job_days',featurey)
    #     Pivot(fea,'list_hour',featurey)
    #     Pivot(fea,'list_notify',featurey)
    #     Pivot(fea,'app_active',featurey)
    #
    #
    #
    #
    #
    #
    #     print '--------------MODELLING---------'
    #
    #     Classification(fea,final_features,featurey,False)

