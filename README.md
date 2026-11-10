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

6.Compare the graphs and hence we obtained the linear regression for the given datas


## Program:
/*
Program to implement the simple linear regression model for predicting the marks scored.

Developed by: lavanya G

RegisterNumber: 212225230148

*/
```
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

Dataset

<img width="212" height="729" alt="image" src="https://github.com/user-attachments/assets/82e0591f-690e-4c1a-8a06-00286f6ab184" />


Head

<img width="184" height="156" alt="image" src="https://github.com/user-attachments/assets/3308f802-7919-45be-96f7-95e3633ae0dc" />


tail values

<img width="194" height="151" alt="image" src="https://github.com/user-attachments/assets/a0aae29e-f71f-4688-b185-ea0739205834" />


x and y values
<img width="753" height="626" alt="image" src="https://github.com/user-attachments/assets/a671386e-a1fd-4b1b-90ac-57a1aa653246" />


Predication values of X and Y
<img width="754" height="74" alt="image" src="https://github.com/user-attachments/assets/271c2579-1d86-4125-a83b-bd88662f8411" />


MSE,MAE and RMSE
<img width="288" height="94" alt="image" src="https://github.com/user-attachments/assets/89d63972-71eb-4cef-a296-ae5e5964902c" />


Training Set
<img width="531" height="393" alt="image" src="https://github.com/user-attachments/assets/632c6bd5-414a-4cef-9664-04f200ddff63" />


Training Set
<img width="547" height="381" alt="image" src="https://github.com/user-attachments/assets/3dfb34a0-b44f-4b77-80a4-0cf2d64bd817" />

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
