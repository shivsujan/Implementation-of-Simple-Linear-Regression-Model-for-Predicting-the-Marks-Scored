# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: S R Shiv Sujan
RegisterNumber:212223040194
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

```


## Output:

![307237070-7c8d0f44-399a-4a52-9109-be17c60e642f](https://github.com/user-attachments/assets/16b87f3e-43cb-41da-b42d-6baf6ad64fa1)

![307237103-29810b1e-0781-4338-a6ba-b7b8e2d30316](https://github.com/user-attachments/assets/5210ac75-2294-4c1a-8300-d60c5fc15495)

![307237152-7509ecba-284d-4c04-86a7-2932b5e439af](https://github.com/user-attachments/assets/88718607-0fb4-4670-9d78-1853a963eced)

![307237178-6e3d9021-fbe2-421f-a1bc-2902ccd99ff5](https://github.com/user-attachments/assets/0acd20fa-03b0-4cbf-8269-e8e35a798ea1)

![307237218-6574b4f6-3e55-4ff6-9e1e-35b78c86ec66](https://github.com/user-attachments/assets/7f43b928-8a38-4f79-9425-1a22fc530353)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
