import numpy as np
import cupy as cp
import pandas as pd
import re
import os
from datetime import datetime
from ClassifyFun import *

# 数据地址
TIC_Path = r'H:\Breast Cancer Project\yby\TypeClassify\result_data\Tic'
SET_Path = r'H:\Breast Cancer Project\yby\TypeClassify\result_data\Set'
# 分类保存地址
# RESULT_Path = r'H:\Breast Cancer Project\yby\Type13\calculate\version_1'
# RESULT_Path = r'H:\Breast Cancer Project\yby\Type13\calculate\version_2'
# RESULT_Path = r'H:\Breast Cancer Project\yby\Type13\calculate\version_3'
# RESULT_Path = r'H:\Breast Cancer Project\yby\Type13\calculate\version_4'
# RESULT_Path = r'H:\Breast Cancer Project\yby\Type13\calculate\version_5'
RESULT_Path = r'H:\Breast Cancer Project\yby\TypeClassify\Type19\calculate\version_12'


# 阈值调整
# TP_threshold = 90
# F1_threshold = 1.0
# F2_threshold = 0.05
# F3_threshold = 0.08
# lowEn_threshold = 0.1co

F1_threshold = [0.1,0.5,1.0]
F2_threshold = 0.05
F3_threshold = 0.08

if not os.path.exists(RESULT_Path):
    os.makedirs(RESULT_Path)

files = os.listdir(TIC_Path)
# files=["Benign_P007_NOR_7-honghaiting-2019.05.08.nii.gz.csv"]
for i in range(len(files)):
    file = files[i]

    # 调错保存log
    type_err_count = 0
    type_err_log = False
    file_save_Path = os.path.join(RESULT_Path, file)
    # if os.path.exists(file_save_Path):
    #     print(file, ' is exists')
    #     continue

    # 读取数据
    print(file, ' is on processing')
    cp_file = cp.loadtxt(os.path.join(TIC_Path, file), skiprows=1, delimiter=',')

    # 滤波去掉不正常的值
    # maxERlocs = cp.argmax(cp_file[:, :4], axis=1).reshape(-1, 1)
    # cp_filter_file=cp_file[maxERlocs[:,0] != 0]
    Centers = cp_file[:, :-3]  # 最后三位存储位置信息 读TIC曲线时须注意

    rows = cp_file[:, -3].reshape(-1, 1)
    columns = cp_file[:, -2].reshape(-1, 1)
    slices = cp_file[:, -1].reshape(-1, 1)

    maxERlocs = cp.argmax(Centers[:, 1:3], axis=1).reshape(-1, 1)+1
    initalERs = Centers[:, 0].reshape(-1, 1)
    initalERs[initalERs==0]=1

    maxERs = cp.max(Centers[:, 1:3], axis=1).reshape(-1, 1)
    lastERs = Centers[:, -1].reshape(-1, 1)



    # 读取Tp信息
    pd_set = pd.read_csv(os.path.join(SET_Path, file))
    Times = pd_set.iloc[0, 2:]  # 从第二个位置开始是时间
    start_time = pd_set.loc[0, 'time_1']
    startTime = datetime.strptime(start_time, "%H:%M:%S")
    differ_time = []
    for i in range(len(Times)):
        time = Times[i]
        Time = datetime.strptime(time, "%H:%M:%S")
        if i == 0 :
            second = 0
        else:
            second = (Time - startTime).seconds
        differ_time.append(second)

    differ_time = cp.array(differ_time)

    # 计算半定量参数
    # peak_time = [differ_time[maxERlocs[i][0]] for i in range(Centers.shape[0])]
    peak_time = [differ_time[maxERlocs[i][0]]/60 for i in range(Centers.shape[0])]
    peak_time=cp.array([peak_time])
    peak_time=peak_time.T
    washin_stage = (maxERs - initalERs) / (initalERs*peak_time)
    washout_stage = (lastERs - maxERs) / (maxERs)
    stable_washOut = [cp.std(Centers[i, maxERlocs[i][0]:-1] / maxERs[i], ddof=1) for i in range(Centers.shape[0])]
    # low_enhance = [cp.std(Centers[i, :] / initalERs[i], ddof=1) for i in range(Centers.shape[0])]

    F1 = [washIN_stage(washin_stage[i][0],F1_threshold) for i in range(Centers.shape[0])]
    F2 = [washOUT_stage(washout_stage[i][0], F2_threshold) for i in range(Centers.shape[0])]
    F3 = [stable_washOUT(stable_washOut[i], F3_threshold) for i in range(Centers.shape[0])]
    # lowEn = [lowEnhance(low_enhance[i], lowEn_threshold) for i in range(Centers.shape[0])]

    Type = [classify_v3(F1[i], F2[i], F3[i]) for i in range(Centers.shape[0])]

    result_columns = 'rows,columns,slices,washin_stage,washout_stage,stable_washOut,Type'
    fmt = "%d,%d,%d,%f,%f,%f,%d"

    # 分类错误调错
    try:
        cp_result = cp.c_[
            rows, columns, slices, washin_stage, washout_stage, stable_washOut, cp.array(Type)]
    except ValueError as err:
        type_err_log = True
        type_err_count += 1
        print("ValueError: {0}".format(err))
        # 保存错误分类值
        type_error_Path = r'E:\yby\python\Breast_Type19\logs\error_log\type'
        pd_type_err = pd.DataFrame([ F1, F2, F3, Type])
        pd_type_err.T.to_csv(os.path.join(type_error_Path, str(type_err_count) + "_" + file), index=False)
        continue

    # result = np.r_[[result_columns], cp_result.get()]
    # print(os.path.join(RESULT_Path, file))
    cp.savetxt(os.path.join(RESULT_Path, file), cp_result, delimiter=',', fmt=fmt, header=result_columns)
