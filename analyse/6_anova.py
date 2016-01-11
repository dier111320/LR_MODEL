#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ConfigParser
import string, os, sys
import pandas as pd
import numpy as np
import math
import traceback
import time
import json
import statsmodels.api as smi
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import subprocess
import ast
import sendMessage as sM
if os.path.exists(sys.argv[1]+'/bin/run')==False:
	sM.run("get_conf doesn't exist at step 6_anova")
else:
	sys.path.append(sys.argv[1]+'/bin/run')
	import get_conf as gc

def isValid(s):
	return len(s.strip()) > 0 and s != 'null' and s != 'NULL' and s != '0'

def eps(x):
	if x > 100:
		return 1
	if x < -100:
		return 0
	return 1 / (1 + math.exp(-x))

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
		sM.run("error happend when ReadData at step 6_anova")
		sys.stderr.write('error happend when read the data as step 0_datareplace\t%s\n' % line)
		traceback.print_exc(file=sys.stderr)

def GetFeature(confDir): #sys.argv[5]+'/conf/default/'+sys.argv[4]+'/'+sys.argv[2]+'/'+sys.argv[3]+'.feature':为读取数据列为feature的名称 参数5:$Base_Dir/conf/default/$time/$cid/$bid.feature
	if os.path.exists(confDir)==False:
		sM.run(confDir +" doesn't exist when GetFeature at step 6_anova")
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
		sM.run(analysisDir + " doesn't exist at step 6_anova")
	else:
		confp = ConfigParser.ConfigParser() 
		confp.read(analysisDir) #Base_Dir/../conf/data_analysis/$time/$cid/$bid.conf,数据分析之配置文件
		y=confp.get('anova','y')
		return y
'''
def DealData(filename,confDir1,confDir2,Cur_day,cid,bid,analysisDir):
	data=ReadData(filename)
	dataF=pd.DataFrame(data)
	feature, rpv, conf2 = GetConf(confDir1,confDir2,Cur_day,cid,bid)
	dataF.columns=conf2
	obData=dataF[feature]
	data=np.array(dataF) 
	y=GetAnalysisConf(analysisDir)
	y=data[:,y]
	nrow,ncol=obData.shape
	x=np.array(obData).astype(float)
	y=np.array(y).astype(float).reshape(nrow,1)
	dat=np.concatenate((y,x),axis=1)
	datF=pd.DataFrame(dat)
	xnum=datF.shape[1]-1
	xyname=feature
	xyname.insert(0,'label')
	datF.columns=xyname
	xvar='+'.join(feature)
	formula='y~'+xvar
	fit1=ols(formula,datF).fit()
	anova_result=anova_lm(fit1)
	return anova_result
'''
def DealData(filename,confDir1,confDir2,Cur_day,cid,bid,analysisDir):
	if os.path.exists(filename)==False:
		sM.run(filename + " doesn't exist at step 6_anova")
	else:
		data=np.array(ReadData(filename))
		dataF=pd.DataFrame(data)
		feature, conf2 = GetConf(confDir1,confDir2,Cur_day,cid,bid)
		dataF.columns=conf2
		y=GetAnalysisConf(analysisDir)
		y=conf2.index(y)
		y=data[:,y].astype(float)
		x=np.array(dataF[feature]).astype(float)
		#formula='label'+'~'+'+'.join(feature)
		fit1=smi.OLS(y,x).fit()
		#anova_result=anova_lm(fit1)
		return fit1.summary()
	
def OutputResult(filename,confDir1,confDir2,Cur_day,cid,bid,analysisDir):
	fitsummary=DealData(filename,confDir1,confDir2,Cur_day,cid,bid,analysisDir)
	#anova_result.to_csv(resultDir,sep='\t')#resultDir:数据输出路径,sys.argv[1]+'/data/data_analysis/'+tcb+'/anova'
	print fitsummary

if __name__ == "__main__":
	tcb=sys.argv[4]+'/'+sys.argv[2]+'/'+sys.argv[3] #short of time-cid-bid
	filename=sys.argv[1]+'/data/feature_integration/'+tcb+'.null_replaced'
	confDir1=sys.argv[1]+'/conf/default/'+tcb+'.feature'
	confDir2=sys.argv[1]+'/bin'
	Cur_day=sys.argv[4]
	cid=sys.argv[2]
	bid=sys.argv[3]
	analysisDir=sys.argv[1]+'/conf/data_analysis/'+tcb+'.conf'
	#resultDir=sys.argv[1]+'/data/data_analysis/'+tcb+'/anova'
	#resultDir=sys.argv[1]+'/data/data_analysis/'+tcb+'/fitsummary'
	OutputResult(filename,confDir1,confDir2,Cur_day,cid,bid,analysisDir)
