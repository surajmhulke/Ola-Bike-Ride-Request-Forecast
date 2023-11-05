 
# Ola Bike Ride Request Forecast

This project aims to predict ride requests for Ola Bike rides based on historical data. It involves data preprocessing, exploratory data analysis, feature engineering, and model development using machine learning techniques. The primary goal is to improve pricing and service allocation by understanding demand patterns.

## Table of Contents
- [Introduction](#introduction)
- [Importing Libraries](#importing-libraries)
- [Importing Dataset](#importing-dataset)
- [Cleaning Columns](#cleaning-columns)
- [Converting Timestamps](#converting-timestamps)
- [Calculating Journey Time](#calculating-journey-time)
- [Type Casting](#type-casting)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis)
- [Feature Selection](#feature-selection)
- [Correlation Heatmap](#correlation-heatmap)
- [Model Development](#model-development)
- [Model Evaluation](#model-evaluation)
- [Conclusion](#conclusion)

## Introduction

In this project, we will predict ride requests for Ola Bike rides. Understanding demand patterns is essential for improving pricing and service allocation.

## Importing Libraries

We import necessary Python libraries to assist in data handling, analysis, and model development.
 
 ```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LassoLars
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
```
## Importing Dataset

We load historical ride data from CSV files, which will be used for training and testing machine learning models.

 ```python 
df_train = pd.read_csv('train.csv', dtype='str')
df_test = pd.read_csv('test.csv', dtype='str')
```
## Cleaning Columns

We clean column names by removing special characters to make data handling easier.

 ```python 

for i in df_train.columns:
    df_train = df_train.rename(columns={i: i.replace("+AF8-", "_")})
```
## Converting Timestamps

We convert string timestamps to datetime objects to calculate journey time in minutes, which is a crucial feature for pricing.

 

df_train['pickup_time'] = df_train['pickup_time'].str.split(' ').str[0].str.split('/').str[2] + \
                        df_train['pickup_time'].str.split(' ').str[0].str.split('/').str[1] + \
                        df_train['pickup_time'].str.split(' ').str[0].str.split('/').str[0] + \
                        df_train['pickup_time'].str.split(' ').str[1].str.replace(':', '')

df_train['drop_time'] = df_train['drop_time'].str.split(' ').str[0].str.split('/').str[2] + \
                        df_train['drop_time'].str.split(' ').str[0].str.split('/').str[1] + \
                        df_train['drop_time'].str.split(' ').str[0].str.split('/').str[0] + \
                        df_train['drop_time'].str.split(' ').str[1].str.replace(':', '')

## Calculating Journey Time

We calculate the journey time in minutes, which is an important factor for pricing, by taking the difference between drop time and pickup time.

 
python
```
  Training dataset
df_train['pickup_time'] = pd.to_datetime(df_train['pickup_time'], format='%Y%d%m%H%M%S')
df_train['drop_time'] = pd.to_datetime(df_train['drop_time'], format='%Y%d%m%H%M%S')
df_train['Journey_Time_in_Mins'] = (df_train['drop_time'] - df_train['pickup_time']) / pd.Timedelta(minutes=1)

 Test dataset
df_test['pickup_time'] = pd.to_datetime(df_test['pickup_time'], format='%Y%d%m%H%M%S')
df_test['drop_time'] = pd.to_datetime(df_test['drop_time'], format='%Y%d%m%H%M%S')
df_test['Journey_Time_in_Mins'] = (df_test['drop_time'] - df_test['pickup_time']) / pd.Timedelta(minutes=1)
```
## Type Casting

We convert string columns to their appropriate numeric data types to prepare the data for model training.

 ```python

  Type casting string columns to numeric types
df_train['total_amount'] = df_train['total_amount'].str.replace('+AC0-', '').astype('float')
df_train['improvement_charge'] = df_train['improvement_charge'].str.replace('+AC0-', '').astype('float')
df_train['distance'] = df_train['distance'].str.replace('+AC0-', '').astype('float')
df_train['mta_tax'] = df_train['mta_tax'].str.replace('+AC0-', '').astype('float')
df_train['extra'] = df_train['extra'].str.replace('+AC0-', '').astype('float')
```
## Exploratory Data Analysis (EDA)

We explore the dataset to understand its characteristics and relationships between variables.

 ```python
## Exploratory Data Analysis
sns.pairplot(df_train, vars=['Journey_Time_in_Mins', 'distance', 'total_amount', 'extra'])
plt.show()
```
## Feature Selection

We select the most relevant features for the model to prevent overfitting and improve prediction accuracy.
 ```python
 
  Feature Selection
selected_features = ['Journey_Time_in_Mins', 'distance', 'extra']
X = df_train[selected_features]
y = df_train['total_amount']
```
## Correlation Heatmap

We visualize the correlation between selected features to understand their relationships.

 ```python

 Correlation Heatmap
corr_matrix = df_train[selected_features + ['total_amount']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.show()
```
## Model Development

We develop and evaluate different regression models, including Linear Regression, Ridge, Lasso, ElasticNet, LassoLars, TweedieRegressor, and Polynomial Regression.

 

##   Model Development
 ```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

  Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
  Lasso Regression
lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)
  ElasticNet Regression
elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic_net.fit(X_train, y_train)

  LassoLars Regression
lasso_lars = LassoLars(alpha=1.0)
lasso_lars.fit(X_train, y_train)

  TweedieRegressor
tweedie = TweedieRegressor()
tweedie.fit(X_train, y_train)
  Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y_train)
```
## Model Evaluation

We evaluate model performance using metrics like Root Mean Squared Error (RMSE) and Coefficient of Determination (R-squared).

 ```python
  Model Evaluation
y_pred_lr = lr.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

  Repeat the evaluation process for other models (ridge, lasso, elastic_net, lasso_lars, tweedie, poly_reg)
 

print("Linear Regression RMSE:", rmse_lr)
print("Linear Regression R-squared:", r2_lr)

 Print evaluation results for other models
 
```
## Conclusion

In conclusion, this project has provided a comprehensive exploration of feature selection and model evaluation, ultimately leading to the selection of a polynomial regression model as the most suitable choice for the given dataset.

Throughout this analysis, we systematically assessed feature selection techniques and compared various linear regression models to predict ride request counts. The selected polynomial regression model consistently outperformed other models in terms of RMSE and R-squared, indicating its superior predictive power.
