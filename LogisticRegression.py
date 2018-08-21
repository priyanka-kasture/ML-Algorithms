# -*- coding: utf-8 -*-
# Priyanka Kasture | pkasture2010@gmail.com
# Logistic Regression - Machine Learning classification algorithm that is used to predict the probability of a categorical dependent variable.
# Reference Book : Learning Predictive Analytics With Python (https://books.google.co.in/books/about/Learning_Predictive_Analytics_with_Pytho.html?id=Ia5KDAAAQBAJ&printsec=frontcover&source=kp_read_button&redir_esc=y#v=onepage&q&f=false)

# Dataset from: UCI Machine Learning Repository.
# Dataset Link: https://raw.githubusercontent.com/madmashup/targeted-marketing-predictive-engine/master/banking.csv/
# Dataset: Direct marketing campaigns (phone calls) of a Portuguese banking institution.
# The dataset provides the bank customers’ information. It includes 41,188 records and 21 fields.
# Statement: Predict whether the client will subscribe to a term deposit or not.

import pandas as pd # Open source library providing high-performance, easy-to-use data structures and data analysis tools
import matplotlib.pyplot as plt # Python 2D plotting library

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

import seaborn as sns # Data visualization library based on matplotlib
sns.set(style="white")
sns.set(style="whitegrid",color_codes=True)

data = pd.read_csv('C:\Users\hp\Desktop\ucibank.csv',header=0)

# Dataset processing begins.

cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1

cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]

data_final=data[to_keep]
data_final.columns.values

data_final_vars=data_final.columns.values.tolist()
y=['y']
X=[i for i in data_final_vars if i not in y]

logreg = LogisticRegression()

rfe = RFE(logreg, 18) # Recursive Feature Elimination
rfe = rfe.fit(data_final[X], data_final[y] )
# print(rfe.support_)
# print(rfe.ranking_)

# RFE has helped us select the following most significant features.

cols=["previous", "euribor3m", "job_blue-collar", "job_retired", "job_services", "job_student", "default_no", 
      "month_aug", "month_dec", "month_jul", "month_nov", "month_oct", "month_sep", "day_of_week_fri", "day_of_week_wed", 
      "poutcome_failure", "poutcome_nonexistent", "poutcome_success"] 
      
# We will only consider the most significant features in our final dataset.
            
X=data_final[cols]
y=data_final['y']

# Dataset processing ends.
# Fitting begins.

import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test))) # Logistic Regression Model Accuracy.

# Accuracy of logistic regression classifier on test set: 0.90.

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# Cross Validation - Attempts to avoid overfitting. Have used 10-fold Cross-Validation to train the model.

kfold = KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

# 10-fold cross validation average accuracy: 0.897.
# The Cross Validation average accuracy is very close to the Logistic Regression Model Accuracy; the model generalizes well.


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

# 10872+254 correct predictions and 1122+109 incorrect predictions

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

# ROC(Reciever Operating Characteristics) Curve
# The dotted line represents the ROC curve of a purely random classifier.
# A good classifier stays as far away from that line as possible (toward the top-left corner).

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
