# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 18:51:58 2019

@author: Sunny
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import axes3d

#import dataset
data=pd.read_csv("diabetes2.csv")

data.info()

dataset.describe()

#checking for null values
data.isnull().sum().sum()


#data analysis & data visulization

#pairplot
#data_n=dataset[['Glucose','Age','DiabetesPedigreeFunction','BMI','Insulin','SkinThickness','BloodPressure']]
#sns.pairplot(data_n)

sns.boxplot(data.Outcome,data.Glucose)

sns.boxplot(data.Outcome,data.BloodPressure)

sns.boxplot(data.Outcome,data.SkinThickness)

sns.boxplot(data.Outcome,data.Insulin)

sns.boxplot(data.Outcome,data.BMI)

sns.boxplot(data.Outcome,data.DiabetesPedigreeFunction)

sns.boxplot(data.Outcome,data.Age)



data.loc[data['Outcome'] == 0, 'Glucose'].hist()

data.loc[data['Outcome']==1, 'Glucose'].hist()

data.loc[data['Outcome']==0, 'Insulin'].hist()

data.loc[data['Outcome']==1, 'Insulin'].hist()

data.loc[data['Outcome']==1, 'BMI'].hist()

data.loc[data['Outcome']==0, 'BMI'].hist()

data.loc[data['Outcome']==1, 'Age'].hist()

data.loc[data['Outcome']==0, 'Age'].hist()



#checking for true diabetes and false diabetes
truediabetes= dataset.loc[dataset['Outcome']==1]
truediabetes.mean()

falsediabetes= dataset.loc[dataset['Outcome']==0]
falsediabetes.mean()

plt.figure(figsize=(20, 20))
for column_index, column in enumerate(falsediabetes.columns):
    if column == 'Outcome':
        continue
    plt.subplot(4, 4, column_index + 1)
    sns.violinplot(x='Outcome', y=column, data=falsediabetes)


plt.figure(figsize=(20, 20))
for column_index, column in enumerate(truediabetes.columns):
    if column == 'Outcome':
        continue
    plt.subplot(4, 4, column_index + 1)
    sb.violinplot(x='Outcome', y=column, data=truediabetes)



corr=data.corr()
print(corr)

#use of zeros_like function
array = np.arange(10).reshape(5, 2)
mask = np.zeros_like(array,dtype=np.bool)


mask = np.zeros_like(corr, dtype=np.bool)
print(mask)

#Return the indices for the upper-triangle of arr.
mask[np.triu_indices_from(mask)] = True

f,ax=plt.subplots(figsize=(11, 9))


#Make a diverging palette between two HUSL colors.
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,square=True,linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

'''colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(dataset.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)'''

# class distribution
print(" == class distribution ==")
print(data.groupby('Outcome').size())

print(" == Univariate Plots: box and whisker plots. determine outliers = ")
data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
plt.show()
print(" == Univariate Plots: histograms. determine if the distribution is normal-like == ")
data.hist()
plt.show()

import pandas
#from pandas.plotting import scatter_matrix
print("== Multivariate Plots: scatter plot matrix. spot structured relationships between input variables ==")
#scatter_matrix(data)
plt.show()

#*** Logistic Reg
import sklearn
array = data.values
array

X = array[:,0:8] # ivs for train
X
y = array[:,8] # dv
y
test_size = 0.33

#partitioning the data
from sklearn.cross_validation import train_test_split, cross_val_score
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size)
print('Partitioning Done!')

#create the model
regr = skl_lm.LogisticRegression()
regr.fit(X_train, y_train)

#predict the data
pred = regr.predict(X_test)
regr.score(X_test,y_test)

#create the confusion matrix
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
cm_df = pd.DataFrame(confusion_matrix(y_test, pred).T, index=regr.classes_,
columns=regr.classes_)
cm_df.index.name = 'Predicted'
cm_df.columns.name = 'True'
print(cm_df)

print(classification_report(y_test, pred))







































