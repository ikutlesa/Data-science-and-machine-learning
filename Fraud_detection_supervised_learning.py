# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 15:28:45 2020

@author: user
"""


import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
#from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from sklearn.metrics import roc_curve,roc_auc_score, precision_recall_curve, average_precision_score
#from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline # Inorder to avoid testing model on sampled data

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import VotingClassifier



#*****************************************************************************************************************
#                                  Data Exploration - Fraud data analysis
#*****************************************************************************************************************



# Create the training and testing sets
df=pd.read_csv("C:/Users/user/Desktop/Fraud detection/creditcard.csv")
# Explore the features available in your dataframe

print(df.info())
print(df.shape)
print(df.head())
print(df.describe())
print(df.Amount.describe())


# Count the occurrences of fraud and no fraud and print them
is_fraud = df['Class'].value_counts()

# Print the ratio of fraud cases
print((is_fraud / df.Class.count())*100)

# Get the mean for each group
df.groupby('Class').mean()

# Plottingg your data
plt.xlabel("Class")
plt.ylabel("Number of Observations")
is_fraud.plot(kind = 'bar',title = 'Frequency by observation number',rot=0)

# Implement a rule for stating which cases are flagged as fraud
#df['flag_as_fraud'] = np.where(np.logical_and(df.V1 < -3, df.V3 < -5), 1, 0)

# Create a crosstab of flagged fraud cases versus the actual fraud cases
#print(pd.crosstab(df.Class, df.flag_as_fraud, rownames=['Actual Fraud'], colnames=['Flagged Fraud']))

# Plot how fraud and non-fraud cases are scattered 
plt.scatter(df.loc[df['Class'] == 0]['V1'], df.loc[df['Class'] == 0]['V2'], label="Class #0", alpha=0.5, linewidth=0.15)
plt.scatter(df.loc[df['Class'] == 1]['V1'], df.loc[df['Class'] == 1]['V2'], label="Class #1", alpha=0.5, linewidth=0.15,c='r')
plt.show()

# Summarize statistics and see differences between fraud and normal transactions
print(df[df['Class'] == 0].Amount.describe())
print(df[df['Class'] == 1].Amount.describe())

fig, ax = plt.subplots(1, 2, figsize=(18,4))

# Plot the distribution of 'Amount' feature
sns.distplot(df['Amount'].values, ax=ax[1], color='b')
ax[1].set_title('Distribution of Transaction Amount', fontsize=14)
ax[1].set_xlim([min(df['Amount'].values), max(df['Amount'].values)])

# Plot the distribution of 'Time' feature 
sns.distplot(df['Time'].values/(60*60), ax=ax[0], color='r')
ax[0].set_title('Distribution of Transaction Time', fontsize=14)
ax[0].set_xlim([min(df['Time'].values/(60*60)), max(df['Time'].values/(60*60))])
plt.show()


# Plot of high value transactions($200-$2000)
bins = np.linspace(200, 2000, 100)
plt.hist(df[df['Class'] == 0].Amount, bins, alpha=1, density=True, label='Non-Fraud')
plt.hist(df[df['Class'] == 1].Amount, bins, alpha=1, density=True, label='Fraud')
plt.legend(loc='upper right')
plt.title("Amount by percentage of transactions (transactions \$200-$2000)")
plt.xlabel("Transaction amount (USD)")
plt.ylabel("Percentage of transactions (%)")
plt.show()

# Plot of transactions in 48 hours
bins = np.linspace(0, 48, 48) #48 hours
plt.hist((df[df['Class'] == 0].Time/(60*60)), bins, alpha=1, density=True, label='Non-Fraud')
plt.hist((df[df['Class'] == 1].Time/(60*60)), bins, alpha=0.6, density=True, label='Fraud')
plt.legend(loc='upper right')
plt.title("Percentage of transactions by hour")
plt.xlabel("Transaction time from first transaction in the dataset (hours)")
plt.ylabel("Percentage of transactions (%)")
plt.show()

#Transaction Amount vs. Hour
# Plot of transactions in 48 hours
plt.scatter((df[df['Class'] == 0].Time/(60*60)), df[df['Class'] == 0].Amount, alpha=0.6, label='Non-Fraud')
plt.scatter((df[df['Class'] == 1].Time/(60*60)), df[df['Class'] == 1].Amount, alpha=0.9, label='Fraud')
plt.title("Amount of transaction by hour")
plt.xlabel("Transaction time as measured from first transaction in the dataset (hours)")
plt.ylabel('Amount (USD)')
plt.legend(loc='upper right')
plt.show()

# Make a new dataset named "df_scaled" dropping out original "Time" and "Amount"
df_scaled = df.drop(['Time','Amount'],axis = 1,inplace=False)
df_scaled.head()


# Calculate pearson correlation coefficience
corr = df_scaled.corr() 

# Plot heatmap of correlation
f, ax = plt.subplots(1, 1, figsize=(24,20))
sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20})
ax.set_title("Imbalanced Correlation Matrix \n (don't use for reference)", fontsize=24)


#*****************************************************************************************************************
#                                   Data preparaion
#*****************************************************************************************************************

# Define the prep_data function to extrac features 
def prep_data(df):
    X = df.drop(['Class'],axis=1, inplace=False) #  
    X = np.array(X).astype(np.float)
    y = df[['Class']]  
    y = np.array(y).astype(np.float)
    return X,y

# Create X and y from the prep_data function 
X, y = prep_data(df_scaled)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)

#*****************************************************************************************************************
#                         Using Machine Learning classification to catch fraud
#*****************************************************************************************************************

#*****************************************************************************************************************
#                                  Logistic Regression
#*****************************************************************************************************************

# Create the training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)

# Fit a logistic regression model to our data
model = LogisticRegression()
model.fit(X_train, y_train)

# Obtain model predictions
y_predicted = model.predict(X_test)

#Model Evaluation

# Create true and false positive rates
false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_predicted)

# Calculate Area Under the Receiver Operating Characteristic Curve 
probs = model.predict_proba(X_test)
roc_auc = roc_auc_score(y_test, probs[:, 1])
print('ROC AUC Score:',roc_auc)

# Obtain precision and recall 
precision, recall, thresholds = precision_recall_curve(y_test, y_predicted)

# Calculate average precision 
average_precision = average_precision_score(y_test, y_predicted)

# Define a roc_curve function
def plot_roc_curve(false_positive_rate,true_positive_rate,roc_auc):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=5, label='AUC = %0.3f'% roc_auc)
    plt.plot([0,1],[0,1], linewidth=5)
    plt.xlim([-0.01, 1])
    plt.ylim([0, 1.01])
    plt.legend(loc='upper right')
    plt.title('Receiver operating characteristic curve (ROC)')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

# Define a precision_recall_curve function
def plot_pr_curve(recall, precision, average_precision):
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.show()

# Print the classifcation report and confusion matrix
print('Classification report:\n', classification_report(y_test, y_predicted))
print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))

# Plot the roc curve 
plot_roc_curve(false_positive_rate,true_positive_rate,roc_auc)

# Plot recall precision curve
plot_pr_curve(recall, precision, average_precision)


#Logistic Regression with Resampled Data
#Logistic Regression with sampled Data using Pipeline

# Create the training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)

# Define which resampling method and which ML model to use in the pipeline
resampling = BorderlineSMOTE(kind='borderline-2',random_state=0) # instead SMOTE(kind='borderline2') 
model = LogisticRegression() 

# Define the pipeline, tell it to combine SMOTE with the Logistic Regression model
pipeline = Pipeline([('SMOTE', resampling), ('Logistic Regression', model)])

# Fit your pipeline onto your training set and obtain predictions by fitting the model onto the test data 
pipeline.fit(X_train, y_train) 
y_predicted = pipeline.predict(X_test)

# Obtain the results from the classification report and confusion matrix 
print('Classifcation report:\n', classification_report(y_test, y_predicted))
print('Confusion matrix:\n', confusion_matrix(y_true = y_test, y_pred = y_predicted))


#*****************************************************************************************************************
#                                  Decision Tree Classifier
#*****************************************************************************************************************

# Create the training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)

# Fit DecisionTree model to our data
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Obtain model predictions
y_predicted = model.predict(X_test)

# Calculate average precision 
average_precision = average_precision_score(y_test, y_predicted)

# Obtain precision and recall 
precision, recall, _ = precision_recall_curve(y_test, y_predicted)

# Plot the recall precision tradeoff
plot_pr_curve(recall, precision, average_precision)

# Print the classifcation report and confusion matrix
print('Classification report:\n', classification_report(y_test, y_predicted))
print('Confusion matrix:\n',confusion_matrix(y_true = y_test, y_pred = y_predicted))

#Precision = 113/(113+25) = 0.82. The rate of true positive in all positive cases.
#Recall = 113/ (113+34) = 0.77. The rate of true positive in all true cases.
#F1-score = 0.79 False positives cases = 31.


#Decision Tree Classifier with SMOTE Data¶

# Define which resampling method and which ML model to use in the pipeline
resampling = BorderlineSMOTE(kind='borderline-2',random_state=0) # instead SMOTE(kind='borderline2') 
model = DecisionTreeClassifier() 

# Define the pipeline, tell it to combine SMOTE with the DecisionTreeClassifier model
pipeline = Pipeline([('SMOTE', resampling), ('Decision Tree Classifier', model)])

# Fit your pipeline onto your training set and obtain predictions by fitting the model onto the test data 
pipeline.fit(X_train, y_train) 
y_predicted = pipeline.predict(X_test)

# Obtain the results from the classification report and confusion matrix 
print('Classifcation report:\n', classification_report(y_test, y_predicted))
print('Confusion matrix:\n',  confusion_matrix(y_true = y_test, y_pred = y_predicted))

#Precision = 0.63. The rate of true positive in all positive cases.
#Recall = 0.71. The rate of true positive in all true cases.
#F1-score = 0.66
#False positives cases = 62.


#******************************************************************************************************************
#                                   Random Forest Classifier
#******************************************************************************************************************

# Create the training and testing sets
# Split data into training and test set (Create the training and testing sets)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Change the model options
model = RandomForestClassifier(bootstrap=True, 
            # 0: non-fraud , 1:fraud                   
            class_weight={0:1, 1:12}, criterion='entropy',
			
			# Change depth of model
            max_depth=10,
		
			# Change the number of samples in leaf nodes
            min_samples_leaf=10, 

			# Change the number of trees to use
            n_estimators=20, n_jobs=-1, random_state=5)

# Fit your training model to your training set
model.fit(X_train, y_train)

# Obtain the predicted values and probabilities from the model 
y_predicted = model.predict(X_test)

# Predict probabilities
probs = model.predict_proba(X_test)

# Calculate average precision and the PR curve
average_precision = average_precision_score(y_test, y_predicted)

# Obtain precision and recall 
precision, recall, _ = precision_recall_curve(y_test, y_predicted)

# Plot the recall precision tradeoff
plot_pr_curve(recall, precision, average_precision)

# Print the roc auc score, the classification report and confusion matrix
print("auc roc score: ", roc_auc_score(y_test, probs[:,1]))
print('Classifcation report:\n', classification_report(y_test, y_predicted))
print('Confusion matrix:\n', confusion_matrix(y_test, y_predicted))


#Accuracy score = Precision = 0.95. The rate of true positive in all positive cases.
#Recall = 0.73. The rate of true positive in all true cases.
#F1-score = 0.83 False positives cases = 6, which is much better.

#Random Forest Classifier with SMOTE Data Catch Fraud
# Define which resampling method and which ML model to use in the pipeline

resampling = BorderlineSMOTE(kind='borderline-2',random_state=0) # instead SMOTE(kind='borderline2') 
model = RandomForestClassifier() 

# Define the pipeline, tell it to combine SMOTE with the Random Forest Classifier model
pipeline = Pipeline([('SMOTE', resampling), ('Random Forest Classifier', model)])

# Fit your pipeline onto your training set and obtain predictions by fitting the model onto the test data 
pipeline.fit(X_train, y_train) 
y_predicted = pipeline.predict(X_test)

# Predict probabilities
probs = model.predict_proba(X_test)

print(accuracy_score(y_test, y_predicted))
print("AUC ROC score: ", roc_auc_score(y_test, probs[:,1]))
# Obtain the results from the classification report and confusion matrix 

print('Classifcation report:\n', classification_report(y_test, y_predicted))
print('Confusion matrix:\n',  confusion_matrix(y_true = y_test, y_pred = y_predicted))


#***************************************************************************************************************************
#                           Random Forest Classifier model adjustment - GridSearchCV
#***************************************************************************************************************************

#Random Forest Classifier Model adjustments

#GridSearchCV to find optimal parameters for Random Forest Classifier
#With GridSearchCV can be define which performance metric to score the options on. 
#Since for fraud detection we are mostly interested in catching as many fraud cases as possible, 
#we can optimize our model settings to get the best possible Recall score



# Define the parameter sets to test
param_grid = {
    'n_estimators': [1, 30], 
    'max_features': ['auto', 'log2'],  
    'max_depth': [4, 8], 
    'criterion': ['gini', 'entropy']
}

# Define the model to use
model = RandomForestClassifier(random_state=5)

# Combine the parameter sets with the defined model
CV_model = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='recall', n_jobs=-1)

# Fit the model to our training data and obtain best parameters
CV_model.fit(X_train, y_train)

#Out[34]: 
#GridSearchCV(cv=5, estimator=RandomForestClassifier(random_state=5), n_jobs=-1,
#             param_grid={'criterion': ['gini', 'entropy'], 'max_depth': [4, 8],
#                         'max_features': ['auto', 'log2'],
#                         'n_estimators': [1, 30]},
#             scoring='recall')

# Build a RandomForestClassifier using the GridSearchCV parameters
model = RandomForestClassifier(bootstrap=True,
                               class_weight = {0:1,1:12},
                               criterion = 'entropy',
                               n_estimators = 30,
                               max_features = 'auto',
                               min_samples_leaf = 10,
                               max_depth = 8,
                               n_jobs = -1,
                               random_state = 5)

# Fit the model to your training data and get the predicted results
model.fit(X_train,y_train)
y_predicted = model.predict(X_test)

# Calculate average precision 
average_precision = average_precision_score(y_test, y_predicted)

# Obtain precision and recall 
precision, recall, _ = precision_recall_curve(y_test, y_predicted)

# Plot the recall precision tradeoff
plot_pr_curve(recall, precision, average_precision)

# Print the roc_auc_score,Classifcation report and Confusin matrix
probs = model.predict_proba(X_test)

print('roc_auc_score:', roc_auc_score(y_test,probs[:,1]))
print('Classification report:\n',classification_report(y_test,y_predicted))
print('Confusion_matrix:\n',confusion_matrix(y_test,y_predicted))


#*****************************************************************************************************************************
#                                         Voting Classifier
#*****************************************************************************************************************************

#Voting Classifier

#Voting Classifier allows us to improve our fraud detection performance, by combining good aspects from 
#multiple models combining three machine learning models into one, to improve our Random Forest fraud detection model 



# Create the training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)

# Define the three classifiers to use in the ensemble
clf1 = LogisticRegression(class_weight={0:1,1:15},random_state=5)
clf2 = RandomForestClassifier(class_weight={0:1,1:12},
                              criterion='entropy',
                              max_depth=10,
                              max_features='auto',
                              min_samples_leaf=10, 
                              n_estimators=20,
                              n_jobs=-1,
                              random_state=5)
clf3 = DecisionTreeClassifier(class_weight='balanced',random_state=5)

# Combine the classifiers in the ensemble model
ensemble_model = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('dt', clf3)], voting='hard')

# Fit the model to your training data and get the predicted results
ensemble_model.fit(X_train,y_train)
y_predicted = ensemble_model.predict(X_test)

# print roc auc score , Classification report and Confusion matrix of the model
print('Classifier report:\n',classification_report(y_test,y_predicted))
print('Confusion matrix:\n',confusion_matrix(y_test,y_predicted))


#Adjust weights within the Voting Classifier

#We adjust the weights we give to these models. By increasing or decreasing weights we can play with how 
#much emphasis we give to a particular model relative to the rest. 

# Define the ensemble model
ensemble_model = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], 
                                  voting='soft', 
                                  weights=[1, 4, 1], 
                                  flatten_transform=True)

# Fit the model to your training data and get the predicted results
ensemble_model.fit(X_train,y_train)
y_predicted = ensemble_model.predict(X_test)

# Calculate average precision 
average_precision = average_precision_score(y_test, y_predicted)

# Obtain precision and recall 
precision, recall, _ = precision_recall_curve(y_test, y_predicted)

# Plot the recall precision tradeoff
plot_pr_curve(recall, precision, average_precision)

# print roc auc score , Classification report and Confusion matrix of the model
print('Classifier report:\n',classification_report(y_test,y_predicted))
print('Confusion matrix:\n',confusion_matrix(y_test,y_predicted))

#The weight option allows us to play with the individual models to get the best final mix for your fraud detection model.
#But the model performance does not improve.




    