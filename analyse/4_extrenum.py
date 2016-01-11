#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import traceback
import sys, os
import ConfigParser
import json
import heapq
import subprocess
import ast
import sendMessage as sM
if os.path.exists(sys.argv[1]+'/bin/run')==False:
	sM.run("get_conf doesn't exist at step 4_extrenum")
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
		sM.run("error happend when ReadData at step 4_extrenum")
		sys.stderr.write('error happend when read the data as step 4_extrenum\t%s\n' % line)
		traceback.print_exc(file=sys.stderr)

def GetFeature(confDir): #sys.argv[5]+'/conf/default/'+sys.argv[4]+'/'+sys.argv[2]+'/'+sys.argv[3]+'.feature':为读取数据列为feature的名称 参数5:$Base_Dir/conf/default/$time/$cid/$bid.feature
	if os.path.exists(confDir)==False:
		sM.run(confDir +" doesn't exist when GetFeature at step 4_extrenum")
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
		sM.run(analysisDir + " doesn't exist at step 4_extrenum")
	else:
		confp = ConfigParser.ConfigParser() 
		confp.read(analysisDir) #Base_Dir/../conf/data_analysis/$time/$cid/$bid.conf,数据分析之配置文件
		method=confp.get('extrenum','method')
		nstd=confp.getint('extrenum','nstd')
		topn=confp.getint('extrenum','topn')
		return method,nstd,topn

def DealData(filename,confDir1,confDir2,Cur_day,cid,bid,analysisDir):
	if os.path.exists(filename)==False:
		sM.run(filename + " doesn't exist at step 4_extrenum")
	else:
		data=ReadData(filename)
		obData=pd.DataFrame(data)
		feature, conf2 = GetConf(confDir1,confDir2,Cur_day,cid,bid)
		obData.columns=conf2
		nrow,ncol=obData.shape
		coln=len(feature)
		method,nstd,topn=GetAnalysisConf(analysisDir)
		if method == 'First':
			for i in feature:#极值处理,西格玛原则
				ix=obData[i].astype(float)
				imean=ix.mean()
				istd=ix.std()
				imax=imean+nstd*istd
				imin=imean-nstd*istd
				ix[ix>imax]=imax
				ix[ix<imin]=imin
				obData[i]=ix
		elif method == 'Second':
			for i in feature:   #极值处理,均值方法
				ix=obData[i].astype(float)
				imean=ix.mean()
				istd=ix.std()
				imax=imean+nstd*istd
				imin=imean-nstd*istd
				ix[ix>imax]=imean
				ix[ix<imin]=imean
				#ix=ix.replace('NaN',imean)
				obData[i]=ix
		else: #method配置不写成first,second,默认为常规处理
			for i in feature: #常规处理
				ix=pd.DataFrame(obData[i].astype(float))
				topN=int(round((1-topn/100.0)*nrow))
				imax=heapq.nlargest(topN,ix.iloc[:,0])[-1]
				ix[ix>imax]=imax
				obData[i]=ix
		return obData


def OutputResult(filename,confDir1,confDir2,Cur_day,cid,bid,analysisDir,resultDir):
	obData=DealData(filename,confDir1,confDir2,Cur_day,cid,bid,analysisDir)
	obData.to_csv(resultDir,index=False,header=False,sep='\t') #resultDir:sys.argv[1]+'/data/data_analysis/'+tcb+'/extrenum'


if __name__ == "__main__":
	tcb=sys.argv[4]+'/'+sys.argv[2]+'/'+sys.argv[3] #short of time-cid-bid
	filename=sys.argv[1]+'/data/feature_integration/'+tcb+'.null_replaced'
	confDir1=sys.argv[1]+'/conf/default/'+tcb+'.feature'
	confDir2=sys.argv[1]+'/bin'
	Cur_day=sys.argv[4]
	cid=sys.argv[2]
	bid=sys.argv[3]
	analysisDir=sys.argv[1]+'/conf/data_analysis/'+tcb+'.conf'
	resultDir=sys.argv[1]+'/data/data_analysis/'+tcb+'/extrenum'
	OutputResult(filename,confDir1,confDir2,Cur_day,cid,bid,analysisDir,resultDir)
