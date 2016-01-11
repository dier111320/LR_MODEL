#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import traceback
import os, sys, json, ast
import ConfigParser
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from matplotlib.backends.backend_pdf import PdfPages
import heapq
import subprocess
import sendMessage as sM
import math
if os.path.exists(sys.argv[1]+'/bin/run')==False:
	sM.run("get_conf doesn't exist at step 3_histogram")
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
		sM.run("error happend when ReadData at step 3_histogram")
		sys.stderr.write('error happend when read the data as step 3_histogram\t%s\n' % line)
		traceback.print_exc(file=sys.stderr)

def GetFeature(confDir): #sys.argv[5]+'/conf/default/'+sys.argv[4]+'/'+sys.argv[2]+'/'+sys.argv[3]+'.feature':为读取数据列为feature的名称 参数5:$Base_Dir/conf/default/$time/$cid/$bid.feature
	if os.path.exists(confDir)==False:
		sM.run(confDir +" doesn't exist when GetFeature at step 3_histogram")
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
		sM.run(analysisDir + " doesn't exist at step 3_histogram")
	else:
		confp = ConfigParser.ConfigParser() 
		confp.read(analysisDir) #Base_Dir/../conf/data_analysis/$time/$cid/$bid.conf,数据分析之配置文件
		logx=confp.getboolean('hist','logx')
		logy=confp.getboolean('hist','logy')
		ctrbins=confp.getint('hist','ctrbins')
		bins=confp.get('hist','bins')
		bins=json.loads('['+bins+']')
		return logx,logy,bins,ctrbins

def CtrPlot(resultDir,data,ctrbin,feature):
	data=pd.DataFrame(data)
	row, col = data.shape
	if col == 2 and type(ctrbin) == int:
		data=data.astype(float)
		data.columns=['feature','label']
		data=data.sort(columns='feature')
		featuredata=data['feature']
		label=data['label']
		hist, edges=np.histogram(range(row),bins=ctrbin)
		edges1=[int(math.ceil(i)) for i in edges]
		ctrx=list(featuredata.iloc[edges1])
		ctrx1=[str(round((ctrx[i]+ctrx[i+1])/2,2)) for i in range(len(ctrx)-1)]
		ctr=[sum(label[edges1[i]:edges1[i+1]])/np.float(hist[i]) for i in range(len(edges1)-1)]
		plt.plot(ctr)
		plt.annotate('sample size: '+str(hist[0]),xy=(ctrbin-6, max(ctr)-0.01))
		plt.xticks(range(ctrbin),ctrx1,rotation=25)
		plt.ylabel('CTR')
		plt.xlabel('Feature: '+feature)
		plt.title('Equal Frenquency Distribution Map')
		plt.savefig(resultDir+'ctr_'+feature+'.png')
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
	plt.clf()


def ConfTol(bini,logx,logy,ctrbins):
	if type(bini) not in [int,list]:
				bins[i]=20
	if logx!=False or logx!=True:
		logx=False
	if logy!=False or logy!=True:
		logy=False
	if type(ctrbins) != int:
		ctrbins=20
	return bini, logx, logy, ctrbins

def OutputResult(filename,confDir1,confDir2,Cur_day,cid,bid,analysisDir,resultDir):
	if os.path.exists(filename)==False:
		sM.run(filename + " doesn't exist at step 3_histogram")
	else:
		data=ReadData(filename)
		obData=pd.DataFrame(data)
		feature, conf2 = GetConf(confDir1,confDir2,Cur_day,cid,bid)
		obData.columns=conf2
		logx,logy,bins,ctrbins=GetAnalysisConf(analysisDir)
		for i in range(len(feature)):
			feature_name=feature[i]
			my_series=pd.DataFrame(obData[[feature_name,'label']])
			bins[i],logx,logy, ctrbin = ConfTol(bins[i],logx,logy,ctrbins)
			HistPlot(resultDir,my_series,bins[i],feature_name,logx,logy)
			CtrPlot(resultDir,my_series,ctrbin,feature_name)

def scp_png(resultDir,Cur_day,cid,bid):
	dir='{0}/{1}/{2}'.format(Cur_day,cid,bid)
	cmd0='ssh bre@192.168.49.81 "cd /opt/bre/ItemWebsite/EagleEye/static/sample_analysis_hist; mkdir -p {0}"'.format(dir)
	cmd1='cd {0}; scp *.png bre@192.168.49.81:/opt/bre/ItemWebsite/EagleEye/static/sample_analysis_hist/{1}'.format(resultDir,dir)
	os.popen(cmd0)
	os.popen(cmd1)

if __name__ == "__main__":
	tcb=sys.argv[4]+'/'+sys.argv[2]+'/'+sys.argv[3] #short of time-cid-bid
	filename=sys.argv[1]+'/data/feature_integration/'+tcb+'.null_replaced'
	confDir1=sys.argv[1]+'/conf/default/'+tcb+'.feature'
	confDir2=sys.argv[1]+'/bin'
	Cur_day=sys.argv[4]
	cid=sys.argv[2]
	bid=sys.argv[3]
	analysisDir=sys.argv[1]+'/conf/data_analysis/'+tcb+'.conf'
	resultDir=sys.argv[1]+'/data/data_analysis/'+tcb+'/'
	OutputResult(filename,confDir1,confDir2,Cur_day,cid,bid,analysisDir,resultDir)
	scp_png(resultDir,Cur_day,cid,bid)

#sys.argv=['0','/opt/bre/rec/feature_project','C17k','3168D0D4_1D1C_13BB_EB2D_55C1B4655782','2015-12-21']
#python 3_histogram.py /opt/bre/rec/feature_project C17k 3168D0D4_1D1C_13BB_EB2D_55C1B4655782 2015-12-22 
