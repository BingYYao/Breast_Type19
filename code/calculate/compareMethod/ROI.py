import cupy as cp
import re
import os
import pandas as pd
import numpy as np
from datetime import datetime

data_Path=r'H:\Breast Cancer Project\yby\TypeClassify\result_data\Tic'
set_Path=r'H:\Breast Cancer Project\yby\TypeClassify\result_data\Set'
save_Path=r'E:\yby\python\Breast_Type19\dataset\compare_method\ROI'

def calIAuc_60(Centers_m,differ_time):
    timeL60=np.argwhere(differ_time>1)[0][0]
    x1,y1=(differ_time[timeL60-1],Centers_m[timeL60-1])
    x2,y2=(differ_time[timeL60],Centers_m[timeL60])
    k=(y2-y1)/(x2-x1)
    b1=y1-k*x1
    b2=y2-k*x2
    b=(b1+b2)/2

    y=b+k*1

    X=[]
    Y=[]
    # print(timeL60)
    # [print(differ_time[i]) for i in range(timeL60)]
    [X.append(differ_time[i]) for i in range(1,timeL60)]
    [Y.append(Centers_m[i]) for i in range(1,timeL60)]

    X.append(1)
    Y.append(y)

    IAUC60=np.trapz(Y,X,dx=0.01)
    return IAUC60

def three_classify(washout_enhance):
    if washout_enhance > 0.1:
        return 'Rise'
    if washout_enhance < -0.1:
        return 'Decline'
    return 'Plateau'

data_list=[]
columns=["No.patient","No.lesionn","Grade","TTP","IAUC60","SI","WIR","WOR","SER","classify"]
files=[file for file in os.listdir(data_Path) if not re.search('txt',file)]
for file in files:
    file_path=os.path.join(data_Path,file)
    cp_data=cp.loadtxt(file_path,skiprows=1, delimiter=',')

    Centers=cp_data[:,:-3]

    Centers_m=cp.average(Centers,axis=0)

    Centers_m=cp.asnumpy(Centers_m)
    Centers_m=Centers_m/Centers_m[0]-1

    # 读取Tp信息
    pd_set = pd.read_csv(os.path.join(set_Path, file))
    Times = pd_set.iloc[0, 2:]  # 从第二个位置开始是时间
    start_time = pd_set.loc[0, 'time_2']
    startTime = datetime.strptime(start_time, "%H:%M:%S")
    differ_time = []
    for i in range(len(Times)):
        time = Times[i]
        Time = datetime.strptime(time, "%H:%M:%S")
        if i == 0 or i == 1:
            second = 0
        else:
            second = (Time - startTime).seconds
        differ_time.append(second)

    differ_time = np.array(differ_time)
    differ_time = differ_time/60

    maxloc=np.argmax(Centers_m[:4])

    # 半定量参数
    TTP=differ_time[maxloc]
    IAUC60=calIAuc_60(Centers_m,differ_time)
    SI=np.max(Centers_m)
    WIR=Centers_m[maxloc]/TTP
    WOR=(Centers_m[-1]-Centers_m[maxloc])/(differ_time[-1]-differ_time[maxloc])
    SER=(Centers_m[maxloc]-Centers_m[0])/(Centers_m[-1]-Centers_m[0])

    # washout 三分类
    washout_enhance = (Centers_m[-1]-Centers_m[maxloc])/Centers_m[maxloc]
    classify=three_classify(washout_enhance)

    # 保存文件
    names_str=file.split('_')

    if names_str[0]=='Benign':
        grade=0
    else:
        grade=1
    patient=names_str[0]+'_'+names_str[1]+'_'+names_str[2]
    label=names_str[3][:-4]

    result=[patient,label,grade,TTP,IAUC60,SI,WIR,WOR,SER,classify]
    data_list.append(result)
    # break


# 查找异常值并替换
np_datalist=np.array(data_list)
np_datalist_semi=np.array(np.array(data_list)[:,3:9],dtype=float)
np_datalist_semi[~np.isfinite(np_datalist_semi)]=0

result=np.c_[np_datalist[:,0:3],np_datalist_semi,np_datalist[:,-1]]

pd_result=pd.DataFrame(result,columns=columns)
pd_result.to_csv(os.path.join(save_Path,'ROI.csv'),index=False)