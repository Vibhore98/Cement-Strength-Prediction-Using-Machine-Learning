import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Importing the data
data =  pd.read_csv('Material Compressive Strength Experimental Data_999be928-cc00-4d78-8bac-9a09dcb9b462.csv')
#Reading the data
data.head()
data.shape
data.columns
data.describe()
data.info()
data.rename(columns={'Material Quantity (gm)': 'material_qty', 'Additive Catalyst (gm)': 'additive_cat', 'Ash Component (gm)': "ash_com",'Water Mix (ml)': 'water_mix', 'Plasticizer (gm)':'platicizer', 'Moderate Aggregator': 'moderate_aggt','Refined Aggregator': 'refined_aggt', 'Formulation Duration (hrs)': 'formulation_durt','Compression Strength MPa': 'compression_sgth'},  inplace=True)
#Checking  null values
data.isnull().sum()
#Removing null values
data.fillna(data.mean(),inplace = True)
data.isnull().sum()
# Checking duplicate values
data.duplicated().sum()
# Removing duplicate values
data.drop_duplicates(inplace=True)
#EDA
for col in data.columns:
    if col != 'compression_sgth':
        sns.scatterplot(x= 'compression_sgth', y=col,data =data )
        plt.show()
from statsmodels.stats.outliers_influence import variance_inflation_factor
def compute_vif(dataset):
    vif = pd.DataFrame()
    vif["Variable"] = dataset.columns
    vif["VIF_Value"] = [variance_inflation_factor(data.values, i) for i in range(dataset.shape[1])]
    return vif
compute_vif(data)
X = data.drop(["compression_sgth",'moderate_aggt','water_mix','refined_aggt'],axis=1)
y = data["compression_sgth"]
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from category_encoders.binary import BinaryEncoder
num_features = [feature for feature in X.columns if X[feature].dtype != 'O']
numeric_transformer = StandardScaler()
bin_transformer = BinaryEncoder()
oh_transformer = OneHotEncoder()
preprocessor = ColumnTransformer([("num", numeric_transformer, num_features)])
from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y = train_test_split(X,y,test_size=0.2,random_state=100)
from sklearn.ensemble import RandomForestRegressor
#create regressor object
RF_regressor = RandomForestRegressor(n_estimators=100,random_state=42)
RF_regressor.fit(train_X,train_y)
import pickle
pickle.dump(RF_regressor,open('RFMODEL.pkl','wb'))

