#!/user/bin/env python
# encoding='utf-8'

import sys, os
import ConfigParser
#import sendMessage as sM
import pandas as pd
import numpy as np
from scipy import stats
import glob
from pandas.io import sql
import MySQLdb
import traceback
import shutil
import threading
if os.path.exists(sys.argv[1]+'/bin/run/sendMessage.py'):
    import sendMessage as sM
else:
    print "sendMessage.py doesn't exist"

if os.path.exists(sys.argv[1]+'/bin/run/get_conf.py')==False:
	sM.run("get_conf doesn't exist at step feature analysis")
else:
	sys.path.append(sys.argv[1]+'/bin/run')
	import get_conf as gc


def sendM(filepath):
	if os.path.exists(filepath)==False:
		sM.run(filepath+'does not exist')
		os.makedirs(filepath)


def PreEnv(Base_Dir, Share_Dir):
	dataDir='/'.join([Share_Dir, 'data', 'feature_integration']) 
	dataAnalysisResult='/'.join([Base_Dir, 'data', 'data_analysis'])
	confDir='/'.join([Base_Dir, 'conf', 'data_analysis'])
	logDir='/'.join([Base_Dir, 'log', 'data_analysis']) 
	for i in [dataDir, dataAnalysisResult, confDir, logDir]:
		sendM(i)
	return  dataDir, dataAnalysisResult, confDir, logDir

def PythonCmd(Share_Dir, cid, bid, Cur_day, Base_Dir, logDir_cidbid, result_dir):
	argv_dir0=' '+' '.join([Share_Dir, cid, bid, Cur_day, Base_Dir])
	argv_dir1=' '+' '.join([Base_Dir, cid, bid, Cur_day])
	cmd0='/opt/Python-2.7.2/bin/python2.7 '+'/'.join([Base_Dir, 'bin/data_analysis/0_datareplace.py'])+argv_dir0 +' 2> '+logDir_cidbid+'datareplace.log'
	cmd1='/opt/Python-2.7.2/bin/python2.7 '+'/'.join([Base_Dir, 'bin/data_analysis/1_threshold.py'])+argv_dir0+' 1> '+result_dir+'threshold.log'+' 2> '+logDir_cidbid+'threshold.log' 
	cmd3='/opt/Python-2.7.2/bin/python2.7 '+'/'.join([Base_Dir, 'bin/data_analysis/3_histogram.py'])+argv_dir1+' 2> '+logDir_cidbid+'histogram.log' 
	cmd4='/opt/Python-2.7.2/bin/python2.7 '+'/'.join([Base_Dir, 'bin/data_analysis/4_extrenum.py'])+argv_dir1+' 2> '+logDir_cidbid+'extrenum.log' 
	cmd5='/opt/Python-2.7.2/bin/python2.7 '+'/'.join([Base_Dir, 'bin/data_analysis/5_correlation.py'])+argv_dir1+ ' 1> '+result_dir+'correlation.log'+' 2> '+logDir_cidbid+'correlation.log' 
	#cmd6='/opt/Python-2.7.2/bin/python2.7 '+'/'.join([Base_Dir, 'bin/data_analysis/6_anova.py'])+argv_dir1+ ' 1> '+result_dir+'fitsummary'+' 2> '+logDir_cidbid+'anova.log'
	cmd6='echo "finish"'
	return cmd0, cmd1, cmd3, cmd4, cmd5, cmd6

def threadT1(cmd1, cmd2):
	t1 = threading.Thread(target=os.popen,args=(cmd1,))
	t2 = threading.Thread(target=os.popen,args=(cmd2,))
	
	t1.start()
	t2.start()
	
	t1.join()
	t2.join()

def threadT2(cmd1, cmd2, cmd3,cmd4):
	t3 = threading.Thread(target=os.popen,args=(cmd1,))
	t4 = threading.Thread(target=os.popen,args=(cmd2,))
	t5 = threading.Thread(target=os.popen,args=(cmd3,))
	t6 = threading.Thread(target=os.popen,args=(cmd4,))
	
	t3.start()
	t4.start()
	t5.start()
	t6.start()
	
	t3.join()
	t4.join()
	t5.join()
	t6.join()


def ErgodicPath(confDir2, Cur_day, Share_Dir, Base_Dir, dataDir, dataAnalysisResult, confDir, logDir, cid, bid):
	filepath=dataDir+'/'+'/'.join([Cur_day, cid, bid])+'.integration' #数据文件
	conf_cid = confDir+'/'+'/'.join([Cur_day, cid])
	confpath=conf_cid+'/'+bid+'.conf'
	if os.path.exists(conf_cid)==False:
		os.makedirs(conf_cid)
	shutil.copyfile('/'.join([confDir,'data_analysis.ini']),confpath)
	if os.path.exists(filepath) and len(open(filepath).readlines())>10:
		#print filepath
		logDir_cidbid='/'.join([logDir, Cur_day, cid, bid])+'/'
		result_dir='/'.join([dataAnalysisResult, Cur_day, cid, bid])+'/'
		if os.path.exists(logDir_cidbid)==False:
			os.makedirs(logDir_cidbid)
		if os.path.exists(result_dir)==False:
			os.makedirs(result_dir)
		cmd0, cmd1, cmd3, cmd4, cmd5, cmd6 = PythonCmd(Share_Dir, cid, bid, Cur_day, Base_Dir, logDir_cidbid, result_dir)
		print cmd0,'\n', cmd1,'\n', cmd3,'\n', cmd4, '\n', cmd5, '\n', cmd6
		Task1=threading.Thread(target=threadT1,args=(cmd0, cmd1,))
		Task1.start()
		Task1.join()
		Task2=threading.Thread(target=threadT2,args=(cmd3, cmd4, cmd5, cmd6,))
		Task2.start()
		Task2.join()
		#thread_cmd = threading.Thread(target=Thread3, args=(cmd0, cmd1, cmd3, cmd4, cmd5, cmd6,))
		#thread_cmd.start()
		#thread_cmd.join()
	else:
		print "{0} doesn't exist or data number is lower 10".format(filepath)

if __name__ == "__main__": 
	Base_Dir=sys.argv[1] 
	#Base_Dir='/opt/bre/rec/feature_project'
	Share_Dir=sys.argv[2]
	#Share_Dir='/opt/share/feature_project'
	Cur_day=sys.argv[3]
	BdmsIp=sys.argv[4] #/opt/bre/bre/xiaoyan.du/feature_project2/bin
	cid=sys.argv[5]
	bid=sys.argv[6]
	confDir2=Base_Dir+'/bin'
	dataDir, dataAnalysisResult, confDir, logDir= PreEnv(Base_Dir, Share_Dir)
	ErgodicPath(confDir2, Cur_day, Share_Dir, Base_Dir, dataDir, dataAnalysisResult, confDir, logDir, cid, bid)
	#data2Sql(confDir2, Cur_day, Share_Dir, Base_Dir, dataDir, dataAnalysisResult, confDir, logDir)
	#data2Hive(dataAnalysisResult,Cur_day,BdmsIp)


'''
sys.argv=['aa', '/opt/bre/rec/feature_project', '/opt/share/feature_project', '2015-11-17', '/opt/bre/bre/xiaoyan.du/feature_project2/bin']
#python data_analysis.py /opt/bre/rec/feature_project /opt/share/feature_project 2015-11-17 /opt/bre/bre/xiaoyan.du/feature_project2/bin
Base_Dir=sys.argv[1] #
#Base_Dir='/opt/bre/rec/feature_project'
Share_Dir=sys.argv[2]
#Share_Dir='/opt/share/feature_project'
Cur_day=sys.argv[3]
BdmsIp=sys.argv[4] #/opt/bre/bre/xiaoyan.du/feature_project2/bin
confDir2=Base_Dir+'/bin'
dataDir, dataAnalysisResult, confDir, logDir= PreEnv(Base_Dir, Share_Dir)
data2Sql(confDir2, Cur_day, Share_Dir, Base_Dir, dataAnalysisResult, confDir, logDir)
data2Hive(dataAnalysisResult,Cur_day,BdmsIp)
if datanum>0:
	data2sql['l_date']=Cur_day
	for i in range(data2sql.shape[0]):
		InsertSql(data2sql.iloc[i,:])
	print "data2sql finished and the number is {0}".format(datanum)
else:
	sM.run("data2sql does not finish in sample analysis")
'''
