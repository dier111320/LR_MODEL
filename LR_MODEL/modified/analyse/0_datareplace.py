#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import traceback
import sys
import ConfigParser
import json
import heapq
import os
import subprocess
import ast
from scipy import stats
import sendMessage as sM
import MySQLdb
if os.path.exists(sys.argv[5]+'/bin/run/get_conf.py')==False:
	sM.run("get_conf doesn't exist at step 0_datareplace")
else:
	sys.path.append(sys.argv[5]+'/bin/run')
	import get_conf as gc

np.set_printoptions(threshold='nan')
pd.options.display.max_rows=None
pd.options.display.max_columns=None

def InsertSql(s):
	ins="INSERT INTO rec_feature_project_data_analysis (feature_name, validcount, covrate, count, mean, std, min, twenty_five, fifty, seventy_five, ninety_eight, max, skew, skewtest, kurtosis, kurtosistest, coef_variation, rang, cid, bid, l_date) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
	conn= MySQLdb.connect(host='192.168.61.73',port = 9000, user='bfdroot',passwd='qianfendian',db ='rec_report')
	cursor = conn.cursor()
	cursor.execute(ins, tuple(s))
	conn.commit()

def Deletesql(l_date, cid, bid):
	sql = 'DELETE FROM rec_feature_project_data_analysis WHERE l_date ='+'"'+l_date+'" and cid='+'"'+cid+'" and bid='+'"'+bid+'"'
	try:
		conn= MySQLdb.connect(host='192.168.61.73',port = 9000, user='bfdroot',passwd='qianfendian',db ='rec_report')
		cursor = conn.cursor()
		cursor.execute(sql)
		conn.commit()
	except:
		conn.rollback()


def ReadData(filename): #为读取原始数据的路径:sys.argv[1]+'/data/feature_integration/'+sys.argv[4]+'/'+sys.argv[2]+'/'+sys.argv[3]+'.integration'
	try:
		file = open(filename) 
		data=[]
		for line in file:
			line=line.strip()
			cols=line.split('\t')
			if len(cols) >1:
				cols=np.array(cols)
				data.append(cols)
		file.close()
		return data
	except:
		sM.run("error happend when ReadData at step 0_datareplace")
		sys.stderr.write('error happend when read the data at step 0_datareplace\t%s\n' % line)
		traceback.print_exc(file=sys.stderr)


def GetFeature(confDir): #sys.argv[5]+'/conf/default/'+sys.argv[4]+'/'+sys.argv[2]+'/'+sys.argv[3]+'.feature':为读取数据列为feature的名称 参数5:$Base_Dir/conf/default/$time/$cid/$bid.feature
	if os.path.exists(confDir)==False:
		sM.run("error happend when GetFeature at step 0_datareplace")
	else:
		feature=open(confDir).readlines()
		feature=feature[0].strip().split(',')
		return feature

def GetConf(confDir1,confDir2,Cur_day,cid,bid): #get_conf.py文件的路径，获取该python产生的字典,confDir:sys.argv[5]+'/bin',Cur_day:sys.argv[4],cid:sys.argv[2],bid:sys.argv[3]
	print confDir2, Cur_day, cid, bid
	conf1=gc.run(confDir2,Cur_day,cid,bid,'6')
	txt=conf1 #参数3为客户名cid，参数4为推荐栏名bid
	title=[] #获取列名
	rpv=[] #获取对应列名的替换值，RePlaceValue
	zero_effect=[]
	for i in txt.keys():
			title.append(i)
			if 'filler' in txt[i]:
				repfiller=float(txt[i]['filler'])
			else:
				repfiller=0.0
			if 'zero_effect' in txt[i]:
				repzero=txt[i]['zero_effect']
			else:
				repzero=False
			rpv.append(repfiller)
			zero_effect.append(repzero)
	#print title, rpv, zero_effect
	conf3=gc.run(confDir2,Cur_day,cid,bid,'3')
	txt3=conf3
	com_name=txt3['General']['common_fields'].split(',')
	feature=GetFeature(confDir1)
	conf2=com_name+feature+['l_date']  #获得所有数据列名
	rpv=[rpv[title.index(feature[i])] for i in range(len(feature))]
	zero_effect=[zero_effect[title.index(feature[i])] for i in range(len(feature))]
	return feature, conf2,  rpv, zero_effect

def scount(series,a):
	cnt=0
	for x in series:
		if x in a:
			cnt=cnt+1
	return cnt

def CovRate(series,signed): #获得覆盖率
	if signed=='False' or signed==False:
		con=scount(series,['NULL','NaN','0',0,'0.0',0.0])
	if signed=='True' or signed==True:
		con=scount(series,['NULL','NaN'])
	return con,np.round(1-float(con)/len(series),6)


def BasicSummary(series,signed):
	series_len = len(series)
	con, rate = CovRate(series,signed)
	basiclist=[series_len-con, rate]
	return pd.Series(basiclist) 

def BasicSummary1(series):
	series_len = len(series)
	basiclist=[stats.skew(series), stats.skewtest(series)[1], stats.kurtosis(series),stats.kurtosistest(series)[1],stats.variation(series)]
	return np.round(pd.Series(basiclist),decimals=6)


def DealData(filename,confDir1,confDir2,Cur_day,cid,bid):
	if os.path.exists(filename)==False:
		sM.run(filename + " doesn't exist at step 0_datareplace")
	else:
		data=ReadData(filename)
		obData=pd.DataFrame(data)
		feature, conf2,  rpv, zero_effect=GetConf(confDir1,confDir2,Cur_day,cid,bid)
		#print feature, conf2
		obData.columns=conf2
		nrow,ncol=obData.shape
		#print obData[:4]
		data_basic_summary=pd.DataFrame(0,index=np.arange(16),columns=feature)
		for i in range(len(feature)):
			my_series=obData[feature[i]]
			datasummary = BasicSummary(my_series,zero_effect[i])
			my_series=my_series.replace('NULL',rpv[i])
			my_series=my_series.replace('NaN',rpv[i])
			obData[feature[i]]=my_series
			my_series=my_series.astype(float)
			datasummary1 =my_series.describe(percentiles=[.25, .5, .75,.98]).append(BasicSummary1(my_series))
			data_basic_summary.iloc[:,i]=np.array(datasummary.append(datasummary1))
		data_basic_summary.index=('validcount','covrate','count','mean','std','min','25%','50%','75%','98%','max','skew','skew','kurtosis','kurtosistest','coefficient of variation')
		data_basic_summary.loc['range']=data_basic_summary.loc['max']-data_basic_summary.loc['min']
		data_basic_summary.loc['cid']=cid
		data_basic_summary.loc['bid']=bid
		return obData,data_basic_summary.T

def data2Sql(Cur_day, cid, bid, filepath):
	Deletesql(Cur_day, cid, bid)
	if os.stat(filepath).st_size !=0:
		data2sql=pd.read_table(filepath,sep='\t',header=None)
		datanum=len(data2sql)
		data2sql['l_date']=Cur_day
		for i in range(datanum):
			InsertSql(data2sql.iloc[i,:])
		print "data2sql finished and the number is {0}".format(datanum)
	#else:
		#sM.run("data2sql does not finish in sample analysis")

def data2Hive(dataAnalysisResult,Cur_day,BdmsIp):
	cmd1='ssh bre@192.168.44.40 \"mkdir -p {0}/../data/data_analysis/{1}\"'.format(BdmsIp,Cur_day)
	cmd2='scp {0}/{1}/data_basic_summary_forHive bre@192.168.44.40:{2}/../data/data_analysis/{1}'.format(dataAnalysisResult,Cur_day,BdmsIp)
	os.popen(cmd1)
	os.popen(cmd2)

def OutputResult(filename,confDir1,confDir2,Cur_day,cid,bid,resultDir,resultDir1): #resultDir:sys.argv[5]+'/data/data_analysis/'+sys.argv[4]+'/'+sys.argv[2]+'/'+sys.argv[3]+'/datareplace',参数7为结果数据输出路径 
	obData,covratel=DealData(filename,confDir1,confDir2,Cur_day,cid,bid)
	obData=pd.DataFrame(obData)
	if os.path.exists(resultDir.rsplit('/',1)[0])==False:
		os.makedirs(resultDir.rsplit('/',1)[0])
	obData.to_csv(resultDir,sep='\t',header=False,index=False)
	covratel.to_csv(resultDir1,sep='\t',header=False)
	data2Sql(Cur_day, cid, bid, resultDir1)


if __name__ == "__main__":
	tcb=sys.argv[4]+'/'+sys.argv[2]+'/'+sys.argv[3] #short of time-cid-bid
	filename=sys.argv[1]+'/data/feature_integration/'+tcb+'.integration'
	confDir1=sys.argv[5]+'/conf/default/'+tcb+'.feature'
	confDir2=sys.argv[5]+'/bin'
	Cur_day=sys.argv[4]
	cid=sys.argv[2]
	bid=sys.argv[3]
	resultDir=sys.argv[5]+'/data/feature_integration/'+tcb+'.null_replaced'
	resultDir1=sys.argv[5]+'/data/data_analysis/'+tcb+'/data_basic_summary'
	OutputResult(filename,confDir1,confDir2,Cur_day,cid,bid,resultDir,resultDir1)
#python 0_datereplace_v1.py /opt/share/feature_project Cdianyingwang 808F4A2B_27DF_745C_FE59_FC20213054E3 2015-11-23 /opt/bre/rec/feature_project
