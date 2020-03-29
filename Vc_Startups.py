# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 19:57:29 2019

@author: Sunny
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import the dataset
dataset=pd.read_csv("VC_Startups.csv")

#visulaize the dataset
X=dataset.iloc[:,0]
y=dataset.iloc[:,4]

plt.scatter(X,y,color='red',s=50)
plt.xlabel("R&D")
plt.ylabel("Profit")
plt.legend()
plt.show()

X=dataset.iloc[:,1]
y=dataset.iloc[:,4]

plt.scatter(X,y,color='yellow',s=50)
plt.xlabel("Administration")
plt.ylabel("Profit")
plt.legend()
plt.show()

X=dataset.iloc[:,2]
y=dataset.iloc[:,4]

plt.scatter(X,y,color='green',s=50)
plt.xlabel("Marketing-spend")
plt.ylabel("Profit")
plt.legend()
plt.show()

X=dataset.iloc[:,3]
y=dataset.iloc[:,4]

plt.scatter(X,y,color='black',s=50)
plt.xlabel("State")
plt.ylabel("Profit")
plt.legend()
plt.show()

#split the dataset into dependent and independent var
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,4]

#one-hot encoding
states=pd.get_dummies(dataset['State'],drop_first=True)
X=X.drop('State',axis=1)
X=pd.concat([X,state],axis=1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#model creation
from sklearn.linear_model import LinearRegression

model=LinearRegression()

model.fit(x_train,y_train)

#predict the result
y_predict=model.predict(x_test)

from sklearn.metrics import r2_score

#goodness of fit
score=r2_score(y_test,y_predict)
score

#print intercept and coef
print(model.intercept_)
print(model.coef_)

#summary of the model
import statsmodels.formula.api as sm
from statsmodels.api import OLS
OLS(y,X).fit().summary()


