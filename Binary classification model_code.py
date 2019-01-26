import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data=pd.read_csv("C:/Test project/toy_dataset.csv", header=0)
data.head()
data.tail()
data=data.dropna()
print(data.shape)
print(list(data.columns))
print(data.columns.values)
df=data.drop(columns='Number') #dropping column Number
data=df
cat_columns = df.select_dtypes(['category']).columns
print("\ncat_columns")

cat_vars=['Gender','City']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1
#cat_vars=['Illness','Gender','City']
print("cat_vars\n")
print(cat_vars)
print("\ndata.colums")
print(list(data.columns))
print("\ndata.columns.values")
print(data.columns.values)
data_vars=data.columns.values.tolist()
print("\ndata_vars")
print(data_vars) 

to_keep=[i for i in data_vars if i not in cat_vars]
print("\nto_keep")
print(to_keep)

data_final=data[to_keep]
#print(data_final)
#data_final=data_final.dropna()
print("\ndata_final.shape")
print(data_final.shape)
print("\ndata_final.column")
print(list(data_final.columns))
print("\ndata_final.column.values")
#data_final.drop(columns='Number')
print(data_final.columns.values) 

X=data_final.loc[:, data_final.columns != 'Illness']
y = data_final.loc[:, data_final.columns == 'Illness']
print("\nX.shape")
print(X.shape)
print("\nX.columns.values")
print(X.columns.values)
print("\ny.shape")
print(y.shape)
print("\ny.columns.values")
print(y.columns.values)

i = 1
myarray = np.asarray(y)
print(myarray[0])
len_array=len(myarray)
print(len_array)
for i in range (len_array):
    #while i < len(myarray):
    #print (myarray[i])
    if myarray[i] == "Yes":
       myarray[i] =  int(1)
    else:
       myarray[i] = int(0)
    #print (myarray[i])  
    
y = myarray.astype('int')
y.tolist
print(y) 

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns
print("\ncolumns.values")
print(columns.values)
print("\ny_train.shape")
print(y_train.shape)

y_train=np.array(y_train)
print(y_train)
y_train = y_train.ravel()
print("\ny_train.ravel()")
print(y_train)

os_data_X,os_data_y=os.fit_sample(X_train, y_train)
print("\nos_data_y")
print(os_data_y)
print("\nos_data_X")
print(os_data_X)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
print("\nos_data_X")
print(os_data_X)
os_data_y= pd.DataFrame(data=os_data_y,columns=['Illness'])
os_data_y_log = os_data_y
print("\nos_data_y")
print(os_data_y)
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['Illness']==0]))
print("Number of subscription",len(os_data_y[os_data_y['Illness']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['Illness']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['Illness']==1])/len(os_data_X))

print("\nos_data_y_log['Illness']")
print(os_data_y_log['Illness'])

#Recursive Feature Elimination
data_final_vars=data_final.columns.values.tolist()
#y=['Illness']
#print(y)
X=[i for i in data_final_vars if i not in y]
print(X)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
rfe = RFE(logreg, 20)
os_data_y_array=np.array(os_data_y)
os_data_y = os_data_y_array.astype('int')
os_data_y=os_data_y_array.ravel()
os_data_y.tolist
 #Y=Y.astype(‘int’)
print(os_data_y)
rfe = rfe.fit(os_data_X, os_data_y)
print(rfe.support_)
print(rfe.ranking_)

#Implementing the model
cols=['Age', 'Income', 'Gender_Female', 'Gender_Male', 'City_Austin', 'City_Boston', 'City_Dallas', 'City_Los Angeles', 
      'City_Mountain View', 'City_New York City', 'City_San Diego', 'City_Washington D.C.']
X=os_data_X[cols]
print(X)
print(os_data_y)
y=os_data_y_log['Illness']
print(y.dtype)
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())

#The p-values for all variables, except variable Age, are greater  than 0.05 
cols=['Age']
X=os_data_X[cols]
y=os_data_y_log['Illness']

#Logistic Regression Model Fitting
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

#Predicting the test set results and calculating the accuracy

y_pred = logreg.predict(X_test)
print("\ny_pred:",y_pred)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

#Accuracy of logistic regression classifier on test set: 0.49

#Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

#The result is telling us that we have 17697+10861 correct predictions and 18237+11092 incorrect predictions.
#confusion_matrix = confusion_matrix(y_test, y_pred)

#Compute precision, recall, F-measure and support

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


#ROC CURVE
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

