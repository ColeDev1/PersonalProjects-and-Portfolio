import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#import matplotlib.pyplot as plt
import random


kaggleData = pd.read_csv("train.csv")
kaggleTestData = pd.read_csv("test.csv")


kaggleData = kaggleData.fillna(0)

Y_data = kaggleData[["SalePrice"]]
X_data = kaggleData.drop(columns="SalePrice")

X_data = pd.get_dummies(X_data, drop_first=True)

kaggleTestData = kaggleTestData.fillna(0)
print(kaggleTestData.head())
kaggleTestData  = pd.get_dummies(kaggleTestData, drop_first=True)
print(kaggleTestData.head())
modelA = LinearRegression()
modelA.fit(X_data,Y_data)

differences = []
for i in range(30):
    index = random.randint(1,1000)
    TestData = kaggleTestData.iloc[index]

    output = modelA.predict(TestData)
    print(output)



#print(kaggleData.head())
# print(kaggleData.isnull().sum())
print("No Errors!")