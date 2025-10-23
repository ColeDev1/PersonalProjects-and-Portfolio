import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import random


kaggleData = pd.read_csv("train.csv")
kaggleTestData = pd.read_csv("test.csv")



kaggleData = kaggleData.fillna(0)

Y_data = kaggleData["SalePrice"]
X_data = kaggleData.drop(columns="SalePrice")

kaggleTestData = kaggleTestData.fillna(0)

modelA = LinearRegression()
modelA.fit(X_data,Y_data)

for i in range(30):
    index = random.randint(1,1000)
    TestData = kaggleTestData.iloc[index]
    testPrice = TestData["SalePrice"]
    testX = TestData.drop["SalePrice"]



#print(kaggleData.head())
# print(kaggleData.isnull().sum())
print("No Errors!")