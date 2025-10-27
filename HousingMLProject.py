import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import csv

#open files
kaggleData = pd.read_csv("train.csv")
kaggleTestData = pd.read_csv("test.csv")

Y_data = kaggleData[["SalePrice"]]
kaggleData = kaggleData.drop(columns=["SalePrice","Id"])

str_data_cols = (kaggleData.select_dtypes(include=['object', 'string']).columns)
num_data_cols = (kaggleData.select_dtypes(exclude=['object', 'string']).columns)

#(name,encoder,columns)
processor = ColumnTransformer([("StringConverter",OneHotEncoder(drop=None, sparse_output=False),str_data_cols),
                               ("NumConverter", StandardScaler(), num_data_cols)])

fixedTrainingData = processor.fit_transform(kaggleData)  
fixedTestingData = processor.fit_transform(kaggleTestData)


# #clean training data
# kaggleData = kaggleData.fillna(0)

# #retrieve prediction data type
# #remove data for training
# X_data = kaggleData.drop(columns=["SalePrice","Id"])
# #set text data to numerical data for model,
# #ex Street could be paved or gravel, the model wants numbers not text.
# X_data = pd.get_dummies(X_data)


# kaggleTestData = kaggleTestData.fillna(0)
# kaggleTestData  = pd.get_dummies(kaggleTestData)
# testDataNoID = kaggleTestData.drop(columns="Id")
# print(str(set(X_data.columns)-set(kaggleTestData.columns)))
# for col in set(set(X_data.columns)-set(kaggleTestData.columns)):
#     kaggleTestData[str(col)] = False



# modelA = LinearRegression() #simplest model to start
# modelA.fit(X_data,Y_data)

# with open('LR_submission.csv', mode="w") as file:
#     file.write("Id, SalePrice\n")
#     for i in range(len(kaggleTestData)):
#         predicted_Id = kaggleTestData.iloc[i+1].loc["Id"]
#         inputData = testDataNoID.iloc[[i+1]]
#         outputData = modelA.predict(inputData)
#         file.write(str(outputData))
#         file.write(",")
#         file.write(str(predicted_Id))
#         file.write('\n')

# # print("No Errors!")