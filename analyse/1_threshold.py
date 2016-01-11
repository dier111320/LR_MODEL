#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import traceback
import ConfigParser
import json
import heapq
import subprocess
import ast
import os
import sys
import sendMessage as sM
if os.path.exists(sys.argv[5]+'/bin/run')==False:
	sM.run("get_conf doesn't exist at step 1_threshold")
else:
	sys.path.append(sys.argv[5]+'/bin/run')
	import get_conf as gc

np.set_printoptions(threshold = 'nan')
pd.set_option('display.max_rows',None)

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
		sM.run("error happend when ReadData at step 1_threshold")
		sys.stderr.write('error happend when read the data as step 1_threshold\t%s\n' % line)
		traceback.print_exc(file=sys.stderr)

def GetFeature(confDir): #sys.argv[5]+'/conf/default/'+sys.argv[4]+'/'+sys.argv[2]+'/'+sys.argv[3]+'.feature':为读取数据列为feature的名称 参数5:$Base_Dir/conf/default/$time/$cid/$bid.feature
	if os.path.exists(confDir)==False:
		sM.run(confDir +" doesn't exist when GetFeature at step 1_threshold")
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

def GetAnalysisConf(analysisDir): #数据分析配置文件#Base_Dir/../conf/data_analysis/$time/$cid/$bid.conf：sys.argv[5]+'/conf/data_analysis/'+sys.argv[4]+'/'+sys.argv[2]+'/'+sys.argv[3]+'.conf'
	if os.path.exists(filename)==False:
		sM.run(analysisDir + " doesn't exist at step 1_threshold")
	else:
		confp = ConfigParser.ConfigParser() 
		confp.read(analysisDir) 
		d = confp.getfloat('threshold','d')
		return d

def OutputResult(filename,confDir1,confDir2,Cur_day,cid,bid,analysisDir):
	if os.path.exists(filename)==False:
		sM.run(filename + " doesn't exist at step 1_threshold")
	else:
		data=ReadData(filename)
		obData = pd.DataFrame(data)
		feature, conf2 = GetConf(confDir1,confDir2,Cur_day,cid,bid)
		obData.columns = conf2
		nrow,ncol = obData.shape
		d=GetAnalysisConf(analysisDir)
		valindex = nrow*d
		freq_table = pd.DataFrame(-100,index = feature,columns = ['value'])
		for i in range(len(feature)):
			my_series = pd.Series(obData[feature[i]])
			counts = my_series.value_counts()
			for j in range(len(counts)):
				if counts[j] >= valindex:
					freq_table.iloc[i] = counts.index[j]
					break
		D_value=freq_table[freq_table['value'] != -100]
		if len(D_value)>0:
			print 'Feature and value of ' +str(d*100)+'%'+'\n', D_value
		else:
			print 'the feature does not exist which somevalue takes up more than ', str(d*100)+'%'

if __name__ == "__main__":
	tcb=sys.argv[4]+'/'+sys.argv[2]+'/'+sys.argv[3] #short of time-cid-bid
	filename=sys.argv[1]+'/data/feature_integration/'+tcb+'.integration'
	confDir1=sys.argv[5]+'/conf/default/'+tcb+'.feature'
	confDir2=sys.argv[5]+'/bin'
	Cur_day=sys.argv[4]
	cid=sys.argv[2]
	bid=sys.argv[3]
	analysisDir=sys.argv[5]+'/conf/data_analysis/'+tcb+'.conf'
	OutputResult(filename,confDir1,confDir2,Cur_day,cid,bid,analysisDir)
