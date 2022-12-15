import numpy as np
import os
import pandas as pd
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import  OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

data_Path=r'E:\yby\python\Breast_Type19\dataset\summarize\Malignant2'
data_Name='Type19.csv'
tp3_Path=r'E:\yby\python\Breast_Type19\dataset\summarize\Malignant2\TP3.csv'

compare_Path=r'E:\yby\python\Breast_Type19\dataset\summarize\Malignant2'
compare_Name='ROI.csv'

data_roi=pd.read_csv(os.path.join(data_Path,data_Name))
data_type19=pd.read_csv(os.path.join(compare_Path,compare_Name))
data_3tp=pd.read_csv(tp3_Path)
def calculate(confusion):
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    Accuracy = (TP+TN) / float(TP+TN+FN+FP)
    Sensitivity = TP / float(TP+FN)
    Specifity = float(TN / float(TN+FP))
    return Accuracy,Sensitivity,Specifity

data=pd.merge(data_roi,data_type19,on=['No.lesionn','Grade'])
data=pd.merge(data,data_3tp,on=['No.lesionn','Grade'])
data.to_csv(r'E:\yby\python\Breast_Type19\dataset\summarize\Malignant2\sumAll.csv')

ROI_data=data.loc[:,'classify']
td=[]
for d in ROI_data:
    if d=='Rise':
        td.append(0)
    elif d=='Plateau':
        td.append(1)
    else:
        td.append(2)
data['classify']=np.array(td)
ROI_data1 = pd.DataFrame(ROI_data)
enc = OneHotEncoder(sparse = False)
ROI_data1 = enc.fit_transform(ROI_data1).astype('str')

ROI_data1= pd.DataFrame(ROI_data1,columns=['Rise','Plateau','Decline'])

data=data.join(ROI_data1)

se=6
labels=[0,1]
X,test_X,y,test_y=train_test_split(data,data.loc[:,'Grade'],test_size=0.3,random_state=se)
save_pred=pd.DataFrame(test_X)

X=pd.DataFrame(X)
X=X.reset_index(drop=True)
y=pd.DataFrame(y)
y=y.reset_index(drop=True)
seed=455
epchos=10
kf = KFold(n_splits=epchos,shuffle=True,random_state=seed)

seed=477
ada_type19=RandomForestClassifier(n_estimators=100,random_state=seed)
ada_semi=RandomForestClassifier(n_estimators=100,random_state=seed)
ada_ROI=ada_ROI=RandomForestClassifier(n_estimators=100,random_state=seed)
ROITtreeFlag=True

count=0
result_data= pd.DataFrame(columns=['model','No','AUC','Accuracy','Sensitivity','Specifity'])
for train_index, test_index in kf.split(X):
    semi_train_X=X.loc[train_index,"TTP":"SER"]
    semi_train_y=X.loc[train_index,'Grade']
    semi_test_X=X.loc[test_index,"TTP":"SER"]
    semi_test_y=X.loc[test_index,'Grade']

    ada_semi.fit(semi_train_X,semi_train_y)

    type19_train_X=X.loc[train_index,"1_%":"19_%"]
    type19_train_y=X.loc[train_index,'Grade']
    type19_test_X=X.loc[test_index,"1_%":"19_%"]
    type19_test_y=X.loc[test_index,'Grade']

    ada_type19.fit(type19_train_X,type19_train_y)

    ROI_train_X=X.loc[train_index,['classify']]
    ROI_train_y=X.loc[train_index,'Grade']
    ROI_test_X=X.loc[test_index,['classify']]
    ROI_test_y=X.loc[test_index,'Grade']

    if ROITtreeFlag:
        ada_ROI.fit(ROI_train_X,ROI_train_y)
        ROITtreeFlag=False

semi_train_X=test_X.loc[:,"TTP":"SER"]
semi_train_y=test_X.loc[:,'Grade']
semi_test_X=test_X.loc[:,"TTP":"SER"]
semi_test_y=test_X.loc[:,'Grade']


y_pre = ada_semi.predict(semi_test_X)
y_prob = ada_semi.predict_proba(semi_test_X)
save_pred['SEMI_pre']=y_pre
for i in range(y_prob.shape[1]):
    head='SEMI_prob_'+str(i)
    save_pred[head]=y_prob[:,i]

confusion = metrics.confusion_matrix(semi_test_y, y_pre,)
Accuracy,Sensitivity,Specifity = calculate(confusion)
AUC = roc_auc_score(semi_test_y, y_prob[:,1])
print('the semi method on ada ----> , Accuracy: ',Accuracy,', Sensitivity: ',Sensitivity,', Specifity: ',Specifity,', AUC: ',AUC,'.')

type19_train_X=test_X.loc[:,"1_%":"19_%"]
type19_train_y=test_X.loc[:,'Grade']
type19_test_X=test_X.loc[:,"1_%":"19_%"]
type19_test_y=test_X.loc[:,'Grade']

y_pre = ada_type19.predict(type19_test_X)
y_prob = ada_type19.predict_proba(type19_test_X)

save_pred['TYPE19_pre']=y_pre
for i in range(y_prob.shape[1]):
    head='TYPE19_prob_'+str(i)
    save_pred[head]=y_prob[:,i]

confusion = metrics.confusion_matrix(type19_test_y, y_pre)
Accuracy,Sensitivity,Specifity = calculate(confusion)
AUC = roc_auc_score(type19_test_y,y_prob[:,1])
print('the Type19 method on ada ----> , Accuracy: ',Accuracy,', Sensitivity: ',Sensitivity,', Specifity: ',Specifity,', AUC: ',AUC,'.')

ROI_train_X=test_X.loc[:,['classify']]
ROI_train_y=test_X.loc[:,'Grade']
ROI_test_X=test_X.loc[:,['classify']]
ROI_test_y=test_X.loc[:,'Grade']

y_pre = ada_ROI.predict(ROI_test_X)
y_prob = ada_ROI.predict_proba(ROI_test_X)

save_pred['ROI_pre']=y_pre
for i in range(y_prob.shape[1]):
    head='ROI_prob_'+str(i)
    save_pred[head]=y_prob[:,i]

confusion = metrics.confusion_matrix(ROI_test_y, y_pre)
Accuracy,Sensitivity,Specifity = calculate(confusion)
AUC = roc_auc_score(ROI_test_y,y_prob[:,1])

print('the roi method on ada ----> , Accuracy: ',Accuracy,', Sensitivity: ',Sensitivity,', Specifity: ',Specifity,', AUC: ',AUC,'.')

save_pred['TRUE']=ROI_test_y