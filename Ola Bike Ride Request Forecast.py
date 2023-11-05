#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from datetime import datetime
df_train = pd.read_csv(r'C:\Users\pc\Desktop\Ola Bike Ride Request Forecast\test.csv', dtype = 'str')
df_test = pd.read_csv(r'C:\Users\pc\Desktop\Ola Bike Ride Request Forecast\train.csv', dtype = 'str')


# Cleaning cols Names

# In[2]:


for i in df_train.columns:
    df_train = df_train.rename(columns = {i:i.replace("+AF8-","_")})


# In[3]:


print(df_train.columns,df_test.columns)


# In[4]:


df_train


# Converting String timestamp to Date timestamp to calculate journey time in Mins as journey time is an important factor for pricing 

# In[ ]:


df_train['pickup_time'] = df_train['pickup_time'].str.split(' ').str[0].str.split('/').str[2]+df_train['pickup_time'].str.split(' ').str[0].str.split('/').str[1]+df_train['pickup_time'].str.split(' ').str[0].str.split('/').str[0]+df_train['pickup_time'].str.split(' ').str[1].str.replace(':','')
df_train['drop_time'] = df_train['drop_time'].str.split(' ').str[0].str.split('/').str[2]+df_train['drop_time'].str.split(' ').str[0].str.split('/').str[1]+df_train['drop_time'].str.split(' ').str[0].str.split('/').str[0]+df_train['drop_time'].str.split(' ').str[1].str.replace(':','')


df_test['pickup_time'] = df_test['pickup_time'].str.split(' ').str[0].str.split('/').str[2]+df_test['pickup_time'].str.split(' ').str[0].str.split('/').str[1]+df_test['pickup_time'].str.split(' ').str[0].str.split('/').str[0]+df_test['pickup_time'].str.split(' ').str[1].str.replace(':','')
df_test['drop_time'] = df_test['drop_time'].str.split(' ').str[0].str.split('/').str[2]+df_test['drop_time'].str.split(' ').str[0].str.split('/').str[1]+df_test['drop_time'].str.split(' ').str[0].str.split('/').str[0]+df_test['drop_time'].str.split(' ').str[1].str.replace(':','')


# Caculating Journey time in Minutes

# In[7]:


# Training dataset
df_train['pickup_time'] = pd.to_datetime(df_train['pickup_time'],format= '%Y%d%m%H%M%S')
df_train['drop_time']  = pd.to_datetime(df_train['drop_time'],format= '%Y%d%m%H%M%S')
df_train['Journer_Time_in_Mins'] = (df_train['drop_time'] - df_train['pickup_time']) / pd.Timedelta(minutes=1)


# Test dataset
df_test['pickup_time'] = pd.to_datetime(df_test['pickup_time'],format= '%Y%d%m%H%M%S')
df_test['drop_time']  = pd.to_datetime(df_test['drop_time'],format= '%Y%d%m%H%M%S')
df_test['Journer_Time_in_Mins'] = (df_test['drop_time'] - df_test['pickup_time']) / pd.Timedelta(minutes=1)


# Type casting string columns to numeric cols for model training

# In[8]:


df_train['total_amount'] = df_train['total_amount'].str.replace('+AC0-','').astype('float')
df_train['improvement_charge'] = df_train['improvement_charge'].str.replace('+AC0-','').astype('float')
df_train['distance'] = df_train['distance'].str.replace('+AC0-','').astype('float')
df_train['mta_tax'] = df_train['mta_tax'].str.replace('+AHs-','').str.replace('+AC0-','').astype('float')
df_train['driver_tip'] = df_train['driver_tip'].str.replace('+AC0-','').astype('float')
df_train['pickup_loc'] = df_train['pickup_loc'].str.replace('+AC0-','').astype('float')
df_train['drop_loc'] = df_train['drop_loc'].str.replace('+AC0-','').astype('float')
df_train['num_passengers'] = df_train['num_passengers'].str.replace('+AC0-','').astype('float')
df_train['extra_charges'] = df_train['extra_charges'].str.replace('+AC0-','').astype('float')
df_train['toll_amount'] = df_train['toll_amount'].str.replace('+AC0-','').astype('float')



df_test['improvement_charge'] = df_test['improvement_charge'].str.replace('+AC0-','').astype('float')
df_test['distance'] = df_test['distance'].str.replace('+AC0-','').astype('float')
df_test['mta_tax'] = df_test['mta_tax'].str.replace('+AHs-','').str.replace('+AC0-','').astype('float')
df_test['driver_tip'] = df_test['driver_tip'].str.replace('+AC0-','').astype('float')
df_test['pickup_loc'] = df_test['pickup_loc'].str.replace('+AC0-','').astype('float')
df_test['drop_loc'] = df_test['drop_loc'].str.replace('+AC0-','').astype('float')
df_test['num_passengers'] = df_test['num_passengers'].str.replace('+AC0-','').astype('float')
df_test['extra_charges'] = df_test['extra_charges'].str.replace('+AC0-','').astype('float')
df_test['toll_amount'] = df_test['toll_amount'].str.replace('+AC0-','').astype('float')


# Doing EDA on dataset

# Selecting Most Relevant features of dataset as we don't want our model to be overfitted by passing too many features. Some of the imp features like demand, traffic, premium locations etc which can and do affect the journey amount were missing in dataset

# In[9]:


df_selec = df_train.loc[:,['distance','mta_tax','toll_amount','extra_charges','improvement_charge','Journer_Time_in_Mins','total_amount']]
df_selec_test = df_test.loc[:,['distance','mta_tax','toll_amount','extra_charges','improvement_charge','Journer_Time_in_Mins']]


# Correlation Heatmap

# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(df_selec.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);
 


# Choosing Regression Model for Prediction as the journey amount are linearly related to selected features.

# Spliting into Train and Validation Dataset

# In[11]:


from sklearn.model_selection import train_test_split

df_selec = df_selec.dropna()

df_selec_y = df_selec['total_amount']
df_selec_x = df_selec.loc[:,['distance','mta_tax','toll_amount','extra_charges','improvement_charge','Journer_Time_in_Mins']]

x_train, x_val, y_train, y_val = train_test_split(df_selec_x, df_selec_y)


# Training the model on Simple Linear Regression

# In[12]:


from sklearn.linear_model import LinearRegression


df_linear_y = y_train
df_linear_x = x_train


linear_reg = LinearRegression()
linear_reg.fit(df_linear_x,df_linear_y)


# Predicting the result for Validation Dataset to determine Model Metrics

# In[13]:


from sklearn.metrics import mean_squared_error, r2_score

y_acc = linear_reg.predict(x_val)

print("Coefficients: \n", linear_reg.coef_)
print("Root Mean squared error: %.2f" % mean_squared_error(y_val, y_acc,squared=False))
print("Coefficient of determination: %.2f" % r2_score(y_val, y_acc))


# Using Ridge Linear Model

# In[14]:


from sklearn.linear_model import Ridge


ridge_reg = Ridge()
ridge_reg.fit(df_linear_x,df_linear_y)

y_acc = ridge_reg.predict(x_val)

print("Coefficients: \n", ridge_reg.coef_)
print("Root Mean squared error: %.2f" % mean_squared_error(y_val, y_acc,squared=False))
print("Coefficient of determination: %.2f" % r2_score(y_val, y_acc))


# Lasso Model

# In[15]:


from sklearn.linear_model import Lasso


lasso_reg = Lasso()
lasso_reg.fit(df_linear_x,df_linear_y)

y_acc = lasso_reg.predict(x_val)

print("Coefficients: \n", lasso_reg.coef_)
print("Root Mean squared error: %.2f" % mean_squared_error(y_val, y_acc,squared=False))
print("Coefficient of determination: %.2f" % r2_score(y_val, y_acc))


# ElasticNet Model

# In[16]:


from sklearn.linear_model import ElasticNet


elasticnet_reg = ElasticNet()
elasticnet_reg.fit(df_linear_x,df_linear_y)

y_acc = elasticnet_reg.predict(x_val)

print("Coefficients: \n", elasticnet_reg.coef_)
print("Root Mean squared error: %.2f" % mean_squared_error(y_val, y_acc,squared=False))
print("Coefficient of determination: %.2f" % r2_score(y_val, y_acc))


# LassoLars Regression

# In[17]:


from sklearn.linear_model import LassoLars


LassoLars_reg = LassoLars()
LassoLars_reg.fit(df_linear_x,df_linear_y)

y_acc = LassoLars_reg.predict(x_val)

print("Coefficients: \n", LassoLars_reg.coef_)
print("Root Mean squared error: %.2f" % mean_squared_error(y_val, y_acc,squared=False))
print("Coefficient of determination: %.2f" % r2_score(y_val, y_acc))


# TweedieRegressor Model

# In[18]:


from sklearn.linear_model import TweedieRegressor


TweedieRegressor_reg = TweedieRegressor()
TweedieRegressor_reg.fit(df_linear_x,df_linear_y)

y_acc = TweedieRegressor_reg.predict(x_val)

print("Coefficients: \n", TweedieRegressor_reg.coef_)
print("Root Mean squared error: %.2f" % mean_squared_error(y_val, y_acc,squared=False))
print("Coefficient of determination: %.2f" % r2_score(y_val, y_acc))


# Polynomial Regression

# In[19]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


polynomial_features= PolynomialFeatures(degree=3)
x_poly = polynomial_features.fit_transform(df_linear_x)

model = LinearRegression()
model.fit(x_poly, df_linear_y)

x_poly = polynomial_features.fit_transform(x_val)
y_acc = model.predict(x_poly)

print("Coefficients: \n", model.coef_)
print("Root Mean squared error: %.2f" % mean_squared_error(y_val, y_acc,squared=False))
print("Coefficient of determination: %.2f" % r2_score(y_val, y_acc))


# Choosing Polynomial Regression model as it yeils minimum RMSE and max R^2 

# predicting values for testing dataset

# In[20]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


polynomial_features= PolynomialFeatures(degree=3)


df_selec_test = df_selec_test.dropna()

x_poly = polynomial_features.fit_transform(df_selec_test)
y_acc = model.predict(x_poly)

df_test_sm = df_selec_test
df_test_sm['total_amount'] = y_acc

df_test_sm


# In conclusion, this machine learning notebook has provided a comprehensive exploration of feature selection and model evaluation, ultimately leading to the selection of a polynomial regression model as the most suitable choice for the given dataset.
# 
# Throughout this analysis, we started by meticulously assessing various feature selection techniques to enhance the model's performance and mitigate overfitting. We then employed multiple linear regression models to predict the target variable, systematically evaluating their accuracy and performance.
# 
# The critical evaluation of model performance, particularly the examination of RMSE (Root Mean Square Error) and R-squared (R^2), revealed that the polynomial regression model consistently outperformed other linear models. This was evidenced by its ability to minimize RMSE, indicating lower prediction errors, and maximize R^2, demonstrating a superior fit to the data.
