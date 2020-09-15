# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 00:01:21 2020

@author: Ravi Rana
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.pandas.set_option('display.max_columns', None)

df = pd.read_csv('./data/train.csv')

print(df.shape)

df.head()

features_with_na = [features for features in df.columns if df[features].isnull().sum()>1]

for feature in features_with_na:
    print(feature, np.round(df[feature].isnull().mean(), 4), '% missing values')



#Because of high number of missing values, we need to find the relationship between missing values and sales price
    
for feature in features_with_na:
    data = df.copy()
    
    #variable which indicates 1 if observation is missing or 0 otherwise
    data[feature] = np.where(data[feature].isnull(), 1, 0)
    
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title(feature)
    plt.show()
    
#NUMERICAL VARIABLES

#list of numerical variables

numerical_features = [feature for feature in df.columns if df[feature].dtypes != 'O']

print('No. of numerical features: ', len(numerical_features))

df[numerical_features].head()

#gaining insight from temporal variables
#in this data we have 4 types of year variables(temporal variables)
#we will analyze how this affects our data

year_feature = [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]

year_feature

#no. of unique years
for feature in year_feature:
    print(feature, df[feature].unique())
    
#we will check the relation between a house and its sales price
    
df.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year Sold')
plt.ylabel('Median House Price')
plt.title('House Price vs Year Sold')

#we can se that as the year sold increases the median house price decreases

year_feature

for feature in year_feature:
    if feature != 'YrSold':
        data = df.copy()
        #we will capture the difference between year variable and the year the house was sold for
        data[feature]=data['YrSold'] - data[feature]
        
        plt.scatter(data[feature], data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show()
        
##Numerical variables are of 2 types
##1. Continous and 2. Discrete

#we will find discrete features
        
discrete_feature = [feature for feature in numerical_features if len(df[feature].unique())<25 and feature not in year_feature+['Id']]
print('discrete features: ' + str(len(discrete_feature)))

discrete_feature

df[discrete_feature].head()

#discrete variable analysis
for feature in discrete_feature:
    data = df.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()
    
##now we follow the same procedure for continous features

continous_features = [feature for feature in numerical_features if feature not in discrete_feature + year_feature + ['Id']]
print('Continous feature count {}' .format(len(continous_features)))

continous_features

#analysis for continous features

for feature in continous_features:
    data = df.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel('count')
    plt.title(feature)
    plt.show()
    
#logarithmic transformation for eliminating skewness
    
for features in continous_features:
    data = df.copy()
    if 0 in data[features].unique():
        pass
    else:
        data[features] = np.log(data[features])
        data['SalePrice'] = np.log(data['SalePrice'])
        plt.scatter(data[features],data['SalePrice'])
        plt.xlabel(features)
        plt.ylabel('SalesPrice')
        plt.title(features)
        plt.show()
        
###Outliers
    

