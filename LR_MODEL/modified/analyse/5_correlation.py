#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import traceback
import sys, os
import ConfigParser
import json
import subprocess
import ast
import sendMessage as sM
import MySQLdb
if os.path.exists(sys.argv[1]+'/bin/run')==False:
	sM.run("get_conf doesn't exist at step 5_correlation")
else:
	sys.path.append(sys.argv[1]+'/bin/run')
	import get_conf as gc

np.set_printoptions(threshold='nan')
pd.options.display.max_rows=None
pd.options.display.max_columns=None


def ReadData(filename): #为读取原始数据的路径:sys.argv[1]+'/data/data_analysis/'+sys.argv[4]+'/'+sys.argv[2]+'/'+sys.argv[3]+'/datareplace'
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
		sM.run("error happend when ReadData at step 5_correlation")
		sys.stderr.write('error happend when read the data as step 0_datareplace\t%s\n' % line)
		traceback.print_exc(file=sys.stderr)

def GetFeature(confDir): #sys.argv[5]+'/conf/default/'+sys.argv[4]+'/'+sys.argv[2]+'/'+sys.argv[3]+'.feature':为读取数据列为feature的名称 参数5:$Base_Dir/conf/default/$time/$cid/$bid.feature
	if os.path.exists(confDir)==False:
		sM.run(confDir +" doesn't exist when GetFeature at step 5_correlation")
	else:
		file2=open(confDir)
		feature=file2.readlines()
		feature=feature[0].strip().split(',')
		file2.close()
		return feature

def GetConf(confDir1,confDir2,Cur_day,cid,bid): #get_conf.py文件的路径，获取该python产生的字典,confDir:sys.argv[5]+'/bin',Cur_day:sys.argv[4],cid:sys.argv[2],bid:sys.argv[3]
	conf1=gc.run(confDir2,Cur_day,cid,bid,'3')
	txt=conf1 #参数3为客户名cid，参数4为推荐栏名bid
	com_name=txt['General']['common_fields'].split(',')
	feature=GetFeature(confDir1)
	conf2=com_name+feature+['l_date']  #获得所有数据列名
	return feature, conf2

def GetAnalysisConf(analysisDir): #数据分析配置文件#Base_Dir/../conf/data_analysis/$time/$cid/$bid.conf，即 sys.argv[1]+'/conf/data_analysis/'+tcb+'.conf'
	if os.path.exists(analysisDir)==False:
		sM.run(analysisDir + " doesn't exist at step 5_correlation")
	else:
		confp = ConfigParser.ConfigParser() 
		confp.read(analysisDir) #Base_Dir/../conf/data_analysis/$time/$cid/$bid.conf,数据分析之配置文件
		rowvar=confp.getint('correlation','rowvar') #If rowvar is non-zero (default), then each row represents a variable, with observations in the columns. 
		bias=confp.getint('correlation','bias') #Default normalization is by (N - 1), where N is the number of observations (unbiased estimate). If bias is 1, then normalization is by N.
		ddof=confp.get('correlation','ddof') #If not None normalization is by (N - ddof), where N is the number of observations;
		corts=confp.getfloat('correlation','corts') #correlation standard
		if ddof == 'None':
			ddof = None
		else:
			ddof = int(ddof)
		return rowvar,bias,ddof,corts


def InsertSql(s):
	ins="INSERT INTO rec_feature_project_feature_correlation (feature_name1, feature_name2, corr_value, cid, bid, l_date) VALUES (%s, %s, %s, %s, %s, %s)"
	conn= MySQLdb.connect(host='192.168.61.73',port = 9000, user='bfdroot',passwd='qianfendian',db ='rec_report')
	cursor = conn.cursor()
	cursor.execute(ins, tuple(s))
	conn.commit()

def Deletesql(l_date, cid, bid):
	sql = 'DELETE FROM rec_feature_project_feature_correlation WHERE l_date ='+'"'+l_date+'" and cid='+'"'+cid+'" and bid='+'"'+bid+'"'
	try:
		conn= MySQLdb.connect(host='192.168.61.73',port = 9000, user='bfdroot',passwd='qianfendian',db ='rec_report')
		cursor = conn.cursor()
		cursor.execute(sql)
		conn.commit()
	except:
		conn.rollback()

def DealData(filename,confDir1,confDir2,Cur_day,cid,bid,analysisDir,resultDir1): #resultDir1:sys.argv[1]+'/data/data_analysis/'+tcb+'/corrxx.csv'
	if os.path.exists(filename)==False:
		sM.run(filename + " doesn't exist at step 5_correlation")
	else:
		data=ReadData(filename)
		obData=pd.DataFrame(data)
		feature, conf2 = GetConf(confDir1,confDir2,Cur_day,cid,bid)
		obData.columns=conf2
		obData=obData[feature]
		obData=np.array(obData).astype(float)
		rowvar,bias,ddof,corts=GetAnalysisConf(analysisDir)
		corr=np.corrcoef(obData, y=None, rowvar=rowvar, bias=bias, ddof=ddof)
		corrdf=pd.DataFrame(corr) #得到相关矩阵
		corrdf.columns=feature
		corrdf.index=feature
		corrdf.to_csv(resultDir1,sep='\t')
		nrow=corrdf.shape[0]
		corrlist=[]
		for i in range(1,nrow):
			for j in range(i):
				corrlist.append([corrdf.index[i],corrdf.columns[j],corrdf.iloc[i,j]])
		corr2sql=pd.DataFrame(corrlist)
		corr2sql.columns=['feature_name1','feature_name2','corr_value']
		corr2sql['cid']=cid
		corr2sql['bid']=bid
		corr2sql['l_date']=Cur_day
		datanum=len(corr2sql)
		Deletesql(Cur_day, cid, bid)
		for i in range(datanum):
			InsertSql(corr2sql.iloc[i,:])
		print "corr2sql finished and the number is {0}".format(datanum)



if __name__ == "__main__":
	tcb=sys.argv[4]+'/'+sys.argv[2]+'/'+sys.argv[3] #short of time-cid-bid
	filename=sys.argv[1]+'/data/feature_integration/'+tcb+'.null_replaced'
	confDir1=sys.argv[1]+'/conf/default/'+tcb+'.feature'
	confDir2=sys.argv[1]+'/bin'
	Cur_day=sys.argv[4]
	cid=sys.argv[2]
	bid=sys.argv[3]
	analysisDir=sys.argv[1]+'/conf/data_analysis/'+tcb+'.conf'
	resultDir1=sys.argv[1]+'/data/data_analysis/'+tcb+'/corrxx.csv'
	DealData(filename,confDir1,confDir2,Cur_day,cid,bid,analysisDir,resultDir1)

#python 5_correlation.py /opt/bre/rec/feature_project C17k 3168D0D4_1D1C_13BB_EB2D_55C1B4655782 2015-12-23

