import numpy as np
import os
import pandas as pd
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import  OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

data_Path=r'E:\yby\python\Breast_Type19\dataset\summarize\molecular_typing'
data_Name='Type19.csv'
tp3_Path=r'E:\yby\python\Breast_Type19\dataset\summarize\molecular_typing\TP3.csv'
compare_Path=r'E:\yby\python\Breast_Type19\dataset\summarize\molecular_typing'
compare_Name='ROI.csv'

data_roi=pd.read_csv(os.path.join(data_Path,data_Name))
data_type19=pd.read_csv(os.path.join(compare_Path,compare_Name))
data_3tp=pd.read_csv(tp3_Path)
def calculate(confusion):
    Classes = confusion.shape[0]
    nums=[]
    for i in range(Classes):
        nums.append(np.sum(confusion[i,1,:]))
    nums=np.array(nums)
    weight_nums=nums/np.sum(nums)
    TP = confusion[:,1, 1]
    TN = confusion[:,0, 0]
    FP = confusion[:,0, 1]
    FN = confusion[:,1, 0]
    Accuracy = (TP+TN) / (TP+TN+FN+FP)
    Sensitivity = TP / (TP+FN)
    Specifity = (TN / (TN+FP))

    weight_Accuracy=0
    weight_Sensitivity=0
    weight_Specifity=0
    for i in range(Classes):
        weight_Accuracy = weight_Accuracy  + weight_nums[i]*Accuracy[i]
    # print(weight_Accuracy)
    for i in range(Classes):
        weight_Sensitivity = weight_Sensitivity  + weight_nums[i]*Sensitivity[i]
    # print(weight_Sensitivity)
    for i in range(Classes):
        weight_Specifity = weight_Specifity  + weight_nums[i]*Specifity[i]
    return weight_Accuracy,weight_Sensitivity,weight_Specifity

data=pd.merge(data_roi,data_type19,on=['No.lesionn','MolecularTypeing'])
data=pd.merge(data,data_3tp,on=['No.lesionn','MolecularTypeing'])
ROI_data=data.loc[:,'classify']
ROI_data = pd.DataFrame(ROI_data)
enc = OneHotEncoder(sparse = False)
ROI_data = enc.fit_transform(ROI_data).astype('str')
ROI_data= pd.DataFrame(ROI_data,columns=['Rise','Plateau','Decline'])
data=data.join(ROI_data)
data['MolecularTypeing']=data['MolecularTypeing']

se=58
labels=['0','1','2','3']
X,test_X,y,test_y=train_test_split(data,data.loc[:,'MolecularTypeing'],test_size=0.3,random_state=se)
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
ada_ROI=RandomForestClassifier(n_estimators=100,random_state=seed)

count=0
result_data= pd.DataFrame(columns=['model','No','AUC','Accuracy','Sensitivity','Specifity'])
for train_index, test_index in kf.split(X):
    semi_train_X=X.loc[train_index,"TTP":"SER"]
    semi_train_y=X.loc[train_index,'MolecularTypeing']
    semi_test_X=X.loc[test_index,"TTP":"SER"]
    semi_test_y=X.loc[test_index,'MolecularTypeing']

    ada_semi.fit(semi_train_X,semi_train_y)

    type19_train_X=X.loc[train_index,"1_%":"19_%"]
    type19_train_y=X.loc[train_index,'MolecularTypeing']
    type19_test_X=X.loc[test_index,"1_%":"19_%"]
    type19_test_y=X.loc[test_index,'MolecularTypeing']

    ada_type19.fit(type19_train_X,type19_train_y)

    ROI_train_X=X.loc[train_index,['Rise','Plateau','Decline']]
    ROI_train_y=X.loc[train_index,'MolecularTypeing']
    ROI_test_X=X.loc[test_index,['Rise','Plateau','Decline']]
    ROI_test_y=X.loc[test_index,'MolecularTypeing']

    ada_ROI.fit(ROI_train_X,ROI_train_y)

semi_train_X=test_X.loc[:,"TTP":"SER"]
semi_train_y=test_X.loc[:,'MolecularTypeing']
semi_test_X=test_X.loc[:,"TTP":"SER"]
semi_test_y=test_X.loc[:,'MolecularTypeing']

y_pre = ada_semi.predict(semi_test_X)
y_prob = ada_semi.predict_proba(semi_test_X)

save_pred['SEMI_pre']=y_pre
for i in range(y_prob.shape[1]):
    head='SEMI_prob_'+str(i)
    save_pred[head]=y_prob[:,i]

confusion = metrics.multilabel_confusion_matrix(semi_test_y, y_pre,labels=[1,2,3,4])
Accuracy,Sensitivity,Specifity = calculate(confusion)
AUC = roc_auc_score(semi_test_y, y_prob, multi_class='ovr')
print('the semi method on ada ----> , Accuracy: ',Accuracy,', Sensitivity: ',Sensitivity,', Specifity: ',Specifity,', AUC: ',AUC,'.')

type19_train_X=test_X.loc[:,"1_%":"19_%"]
type19_train_y=test_X.loc[:,'MolecularTypeing']
type19_test_X=test_X.loc[:,"1_%":"19_%"]
type19_test_y=test_X.loc[:,'MolecularTypeing']

y_pre = ada_type19.predict(type19_test_X)
y_prob = ada_type19.predict_proba(type19_test_X)

save_pred['TYPE19_pre']=y_pre
for i in range(y_prob.shape[1]):
    head='TYPE19_prob_'+str(i)
    save_pred[head]=y_prob[:,i]

confusion = metrics.multilabel_confusion_matrix(type19_test_y, y_pre,labels=[1,2,3,4])
Accuracy,Sensitivity,Specifity = calculate(confusion)
AUC = roc_auc_score(type19_test_y, y_prob, multi_class='ovr')
print('the Type19 method on ada ----> , Accuracy: ',Accuracy,', Sensitivity: ',Sensitivity,', Specifity: ',Specifity,', AUC: ',AUC,'.')

ROI_train_X=test_X.loc[:,['Rise','Plateau','Decline']]
ROI_train_y=test_X.loc[:,'MolecularTypeing']
ROI_test_X=test_X.loc[:,['Rise','Plateau','Decline']]
ROI_test_y=test_X.loc[:,'MolecularTypeing']

y_pre = ada_ROI.predict(ROI_test_X)
y_prob = ada_ROI.predict_proba(ROI_test_X)

save_pred['ROI_pre']=y_pre
for i in range(y_prob.shape[1]):
    head='ROI_prob_'+str(i)
    save_pred[head]=y_prob[:,i]

confusion = metrics.multilabel_confusion_matrix(ROI_test_y, y_pre,labels=[1,2,3,4])
Accuracy,Sensitivity,Specifity = calculate(confusion)
AUC = roc_auc_score(ROI_test_y,  y_prob, multi_class='ovr')

print('the roi method on ada ----> , Accuracy: ',Accuracy,', Sensitivity: ',Sensitivity,', Specifity: ',Specifity,', AUC: ',AUC,'.')