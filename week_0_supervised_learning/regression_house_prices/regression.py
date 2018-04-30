#.. sectionauthor:: Sean Anderson <sean.anderosn@filament.uk.com>

import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn import datasets, linear_model
from sklearn.metrics import r2_score

#Function which reads in csv files.
def read_csv(filename):
    file = pd.read_csv(filename)
    return file

def make_data_numeric(data_name):
    data = data_name
    for category in data_name.columns:
        data[category] = data[category].astype('category')
        data.dtypes
        data[category] = data[category].cat.codes
    return data

#Headers/categorries of the datasets.
headers = ['Id', 'MSSubClass',"MSZoning","LotFrontage","LotArea","Street","Alley","LotShape","LandContour","Utilities","LotConfig","LandSlope","Neighborhood","Condition1","Condition2","BldgType","HouseStyle","OverallQual","OverallCond","YearBuilt","YearRemodAdd","RoofStyle","RoofMatl","Exterior1st","Exterior2nd","MasVnrType","MasVnrArea","ExterQual","ExterCond","Foundation","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinSF1","BsmtFinType2","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","Heating","HeatingQC","CentralAir","Electrical","1stFlrSF","2ndFlrSF","LowQualFinSF","GrLivArea","BsmtFullBath","BsmtHalfBath","FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr","KitchenQual","TotRmsAbvGrd","Functional","Fireplaces","FireplaceQu","GarageType","GarageYrBlt","GarageFinish","GarageCars","GarageArea","GarageQual","GarageCond","PavedDrive","WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","PoolQC","Fence","MiscFeature","MiscVal","MoSold","YrSold","SaleType","SaleCondition","SalePrice"]

#Reads in the training data.
data = read_csv('train.csv')
train_data = data[400:1460]

#Assigns the Y value to be the sale price.
y_values = train_data["SalePrice"]

#Makes the headers of the dataset the headings of the training set.
train_data.columns = headers
train_data = make_data_numeric(train_data)
#Removes the sale price from the training set so that it is not added to the x values.
train_data = train_data.drop("SalePrice", 1)

#Adds x values into an array uses headers[:-1] as it is important that the x values do not contain the sale price.
x_values = []
for header in headers[:-1]:
    x_values.append(train_data[header])

#Builds the regression model.
reg = linear_model.LinearRegression()
reg.fit(np.transpose(np.matrix(x_values)), np.transpose(np.matrix(y_values)))

#Reads in the test data.
test_data = data[1:400]
true_y_values = test_data['SalePrice']
test_data = test_data.drop("SalePrice", 1)
test_data = make_data_numeric(test_data)

#Actually predict the data.
predicted_y_values =  reg.predict(test_data)
r2_score = r2_score(true_y_values, predicted_y_values)
print ("r2 score is:")
print (r2_score)
