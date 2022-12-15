import numpy as np
import cupy as cp
import pandas as pd
import re
import os
from datetime import datetime


# vesrion_1 and version_2
def peakTime_t(peak_time, threshold):
    temp = peak_time
    TP = None
    if temp > threshold:
        TP = 'Low Speed'
    elif temp <= threshold:
        TP = 'High Speed'
    return TP


def enhancementRate_t(enhancement_rate, threshold):
    temp = enhancement_rate
    F1 = None
    if temp > threshold:
        F1 = 'Low Value'
    elif temp <= threshold:
        F1 = 'High Value'
    return F1


def washOutRate_t(washOut_rate, threshold):
    temp = washOut_rate
    F2 = None
    if temp > threshold:
        F2 = 'Rise'
    elif temp <= threshold and temp >= -threshold:
        F2 = 'Plateau'
    elif temp <= -threshold:
        F2 = 'Decline'
    return F2


def stableWASHOUT_t(stable_washOut, threshold):
    temp = stable_washOut
    F3 = None
    if temp > threshold:
        F3 = 'Not Stable'
    else:
        F3 = 'Stable'
    return F3


def lowEnhance(low_enhance, threshold):
    temp = low_enhance
    lowEn = None
    if temp > threshold:
        lowEn = 'Not LowEnhance'
    else:
        lowEn = 'LowEnhance'
    return lowEn


def classify_v1(TP, F1, F2, F3, LowEn):
    Type = None
    if LowEn == 'LowEnhance':
        Type = 19
        return Type
    if TP == 'Low Speed' and F1 == 'Low Value':
        if F2 == 'Rise' and F3 == 'Stable':
            Type = 1
        elif F2 == 'Rise' and F3 == 'Not Stable':
            Type = 2
        elif F2 == 'Plateau' and F3 == 'Stable':
            Type = 3
        elif F2 == 'Plateau' and F3 == 'Not Stable':
            Type = 4
        elif F2 == 'Decline' and F3 == 'Stable':
            Type = 5
        elif F2 == 'Decline' and F3 == 'Not Stable':
            Type = 6
    elif (TP == 'Low Speed' and F1 == 'High Value') or (TP == 'High Speed' and F1 == 'Low Value'):
        if F2 == 'Rise' and F3 == 'Stable':
            Type = 7
        elif F2 == 'Rise' and F3 == 'Not Stable':
            Type = 8
        elif F2 == 'Plateau' and F3 == 'Stable':
            Type = 9
        elif F2 == 'Plateau' and F3 == 'Not Stable':
            Type = 10
        elif F2 == 'Decline' and F3 == 'Stable':
            Type = 11
        elif F2 == 'Decline' and F3 == 'Not Stable':
            Type = 12
    elif TP == 'High Speed' and F1 == 'High Value':
        if F2 == 'Rise' and F3 == 'Stable':
            Type = 13
        elif F2 == 'Rise' and F3 == 'Not Stable':
            Type = 14
        elif F2 == 'Plateau' and F3 == 'Stable':
            Type = 15
        elif F2 == 'Plateau' and F3 == 'Not Stable':
            Type = 16
        elif F2 == 'Decline' and F3 == 'Stable':
            Type = 17
        elif F2 == 'Decline' and F3 == 'Not Stable':
            Type = 18
    return Type


# version_3
def washIN_stage(wash_in_stage, threshold):
    F1 = None
    temp = wash_in_stage
    if temp <= threshold[0]:
        F1 = 'No Enhance'
    elif threshold[0] < temp <= threshold[1]:
        F1 = 'slow'
    elif threshold[1] < temp <= threshold[2]:
        F1 = 'medium'
    elif temp > threshold[2]:
        F1 = 'rapid'
    else:
        print(temp)
    return F1


def washOUT_stage(washOut_stage, threshold):
    temp = washOut_stage
    F2 = None
    if temp > threshold:
        F2 = 'progressive'
    elif threshold >= temp >= -threshold:
        F2 = 'plateau'
    elif temp < -threshold:
        F2 = 'washout'
    return F2


def stable_washOUT(stable_washOut, threshold):
    temp = stable_washOut
    if temp > threshold:
        F3 = 'notstable'
    else:
        F3 = 'stable'
    return F3


def classify_v3(f1c, f2c, f3c):
    typ = None
    if f1c == "slow":
        if f2c == "progressive" and f3c == "stable":
            typ = 1
        elif f2c == "progressive" and f3c == "notstable":
            typ = 2
        elif f2c == "plateau" and f3c == "stable":
            typ = 3
        elif f2c == "plateau" and f3c == "notstable":
            typ = 4
        elif f2c == "washout" and f3c == "stable":
            typ = 5
        elif f2c == "washout" and f3c == "notstable":
            typ = 6
    elif f1c == "medium":
        if f2c == "progressive" and f3c == "stable":
            typ = 7
        elif f2c == "progressive" and f3c == "notstable":
            typ = 8
        elif f2c == "plateau" and f3c == "stable":
            typ = 9
        elif f2c == "plateau" and f3c == "notstable":
            typ = 10
        elif f2c == "washout" and f3c == "stable":
            typ = 11
        elif f2c == "washout" and f3c == "notstable":
            typ = 12
    elif f1c == "rapid":
        if f2c == "progressive" and f3c == "stable":
            typ = 13
        elif f2c == "progressive" and f3c == "notstable":
            typ = 14
        elif f2c == "plateau" and f3c == "stable":
            typ = 15
        elif f2c == "plateau" and f3c == "notstable":
            typ = 16
        elif f2c == "washout" and f3c == "stable":
            typ = 17
        elif f2c == "washout" and f3c == "notstable":
            typ = 18
    elif f1c == "No Enhance":  # necrosis:
        typ = 19

    return typ
