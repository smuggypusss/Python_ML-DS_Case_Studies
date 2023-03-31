# -*- coding: utf-8 -*-
#This is a basic linear regression case study 


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
data_income=pd.read_csv("C:\\Users\\Asus\\Downloads\\income.csv")
data=data_income.copy()
print(data.info())
data.isnull()
print("Data columns with null values",data.isnull().sum())#to check if there is any null values
#Summary of numerical variables
summary_num=data.describe()
print(summary_num)
#Summary of categorical variables
summary_cate=data.describe(include='O')
print(summary_cate)
#Frequency of each categories
data['JobType'].value_counts()
data['occupation'].value_counts()
#Cheacking for unique classes
print(np.unique(data['JobType']))
print(np.unique(data['occupation']))
data=pd.read_csv('C:\\Users\\Asus\\Downloads\\income.csv',na_values=[" ?"])
#
#
#
# DATA PRE-PROCESSING
data.isnull().sum()
missing=data[data.isnull().any(axis=1)]
data2=data.dropna(axis=0) #removing the records for missing values
#Relationship between independent values
correlation=data2.corr()
#Extracting the column names
data2.columns
#Gender proportion table:
gender=pd.crosstab(index= data2['gender'],columns='count',normalize=True)
#doing normalization to find the % of men and women
print(gender)
#Gender vs Salary status
gender_salstat =pd.crosstab(index=data2['gender'], columns=data2['SalStat'],margins=True,normalize='index')
print(gender_salstat)
#Frequency Distribution of Salary Status
SalStat=sns.countplot(data2['SalStat'])
#Histogram for age bins for no. of bars, kde false for getting readings at y-axis
sns.displot(data2['age'],bins=10,kde=False)
#inference- people with age 20-45 are high in numbers
#Box Plot: Age vs Salary Status
sns.boxplot('SalStat','age',data=data2)
data2.groupby('SalStat')['age'].median()

##Logistic Regression
#Classifying the input in two parts
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
#get_dummies makes boolean/binary values for each column ex: job type: federal=0(non federal) 1(federal)
new_data=pd.get_dummies(data2,drop_first=True)
#Storing the column names
column_list=list(new_data.columns)
print(column_list)
#Separating the input names from data
features=list(set(column_list)-set(['SalStat']))
print(features)
#Storing the output values in y
y=new_data['SalStat'].values
print(y)
#Storing the values from input features
x=new_data[features].values
print(x)
#splitting the data into train and test 
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
#Make an instance of the model
Logistic=LogisticRegression()
#fitting the values for x and y
Logistic.fit(train_x,train_y)
Logistic.coef_
Logistic.intercept_
#Prediction for test data
prediction=Logistic.predict(test_x)
print(prediction)
#Confusion Matrix
confusion_matrix=confusion_matrix(test_y,prediction)
print(confusion_matrix)
#Calculating the accuracy
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)
#printing the misclassified values from prediction
print("Missclassified samples: %d" %(test_y!=prediction).sum())


#Logistic Regression Removing Insignificant variables to increase accuracy
cols=['gender','nativecountry','race','JobType']
new_data=data2.drop(cols,axis=1)
new_data=pd.get_dummies(new_data,drop_first=True)
column_list=list(new_data.columns)
features=list(set(column_list)-set(['SalStat']))
y=new_data['SalStat'].values
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
logistic=LogisticRegression()
logistic.fit(train_x,train_y)
prediction=logistic.predict(test_x)
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)
