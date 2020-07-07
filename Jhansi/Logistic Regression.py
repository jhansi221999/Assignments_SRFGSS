#Logistic Regression
import pandas as pd
import numpy as np
import pandas_profiling as pp
df=pd.read_csv("/storage/emulated/0/jhansi/admission1.csv")
print(df.head())
print(df.tail())
print(df.dtypes)
print(df.shape)
print(df.info)
print(df.describe())
Profile = pp.ProfileReport(df)
Profile.to_file("/storage/emulated/0/jhansi/report1.html")
df=df.drop("Research",axis=1)
print(df.shape)
print(df.head())
from sklearn.model_selection import train_test_split
y=df['Chance_of_Admit']
X=df.drop('Chance_of_Admit',axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=1)
#model building
from sklearn.linear_model import LogisticRegression
regression_model=LogisticRegression()
print(regression_model.fit(X_train,y_train))
intercept=regression_model.intercept_[0]
print(intercept)
for idx,col_name in enumerate(X_train.columns):
	print("The co-efficient for {} is {}".format(col_name,regression_model.coef_[0][idx]))
#Evaluation metrics
y_pred=regression_model.predict(X_test)
print(y_pred)
from sklearn.metrics import confusion_matrix,accuracy_score
conf_matrix = confusion_matrix(y_pred,y_test)
print(conf_matrix)
acc_score = accuracy_score(y_pred,y_test)
print(acc_score)
# pre_deployment test
new_parameters=[[340,120,4.5,5.0,5.0,9.8,1]]
Chance_of_Admit_predicted=regression_model.predict(new_parameters)
print(Chance_of_Admit_predicted)
