#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Import libraries
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib 
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (10, 8)


# In[4]:


#create dataframe and store dataset there
# pd to read data
data = pd.read_csv(r'C:\Users\GKamau\Downloads\water_quality_.csv')


# In[6]:


data.head()


# In[7]:


#information about data (missing values and data type)
data.info()


# In[10]:


#column data types
data.dtypes


# In[8]:


#find missing values in all columns - missing value analysis
data.isna().sum()


# In[11]:


#fish out the missing values
data = data.dropna()


# In[12]:


data.isna().sum()


# In[13]:


#descriptive statistics to summarize the central tendency, dispersion and shape of a dataset’s distribution
data.describe()


# In[14]:


#unique value analysis - how many different values e.g(there's 2 entries for check[1 & 0]
dict = {}
for i in list(data.columns):
    dict[i] = data[i].value_counts().shape[0]
pd.DataFrame(dict, index = ["unique count"]).transpose()


# In[15]:


import os


# In[16]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_curve


# In[17]:


get_ipython().system('pip install warnings')


# In[18]:


import warnings


# In[19]:


warnings.filterwarnings("ignore")


# In[20]:


#column analysis - categorical & numerical variables
def grab_col_names(dataframe, cat_th=10):
    #cat_cols
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes =="0"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
    			dataframe[col].dtypes != "0"]
    cat_cols = cat_cols + num_but_cat
   #num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes !="0"]
    num_cols = [col for col in num_cols if col not in cat_cols]
    
    print(f"Variables: {dataframe.shape[1]}")
    print(f"Number of Categorical Variables: {len(cat_cols)}")
    print(f"Number of Numerical Variables: {len(num_cols)}")
        
    return cat_cols, num_cols


# In[21]:


cat_cols, num_cols = grab_col_names(data)


# In[22]:


categorical = print(cat_cols)
Numerical =print(num_cols)


# In[25]:


#Numeric feature analysis
#add output (target variable) to the numeric column list
num_cols.append("output")
num_cols


# In[26]:


#Categorical(cant be counted) Feature analysis
cat_cols


# In[31]:


plt.figure(figsize=(9, 8))
sns.distplot(data['Hardness'], color='g', bins=100, hist_kws={'alpha': 0.4});


# In[38]:


data_num = data.select_dtypes(include = ['float64', 'int64'])
data_num.head()


# In[39]:


data_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);


# In[40]:


#correlation of data using spearman method
data.corr(method = 'spearman')


# In[41]:


#there are no strong correlations apart from a factor and itself 


# In[43]:


#strongly correlated values
data_num_corr = data_num.corr()['Hardness'][:-1]
golden_features_list = data_num_corr[abs(data_num_corr) > 0.5].sort_values(ascending=False)
print("There is {} strongly correlated values with Hardness:\n{}".format(len(golden_features_list), golden_features_list))


# In[44]:


#strongly correlated values
data_num_corr = data_num.corr()['ph'][:-1]
golden_features_list = data_num_corr[abs(data_num_corr) > 0.5].sort_values(ascending=False)
print("There is {} strongly correlated values with ph:\n{}".format(len(golden_features_list), golden_features_list))


# In[48]:


#checking for outliers with scatter plots
for i in range(0, len(data_num.columns), 5):
    sns.pairplot(data=data_num,
                x_vars=data_num.columns[i:i+5],
                y_vars=['ph'])


# In[49]:


#numerical variables distribution
def histPlot(num):
    sns.histplot(data = data, x = num, bins = 50, kde = True)
    print("{} distribution with hist:".format(num))
    plt.show()


# In[50]:


for i in num_cols:
    histPlot(i)


# In[54]:


scaled_array = scaler.fit_transform(data[num_cols[:-1]])
scaled_array


# In[55]:


#collect standardized numeric columns in to a dataframe
data_dummy =pd.DataFrame(scaled_array, columns = num_cols[:-1])
data_dummy.head()


# In[57]:


#train & test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)
print("x_train: {}".format(x_train.shape))
print("x_test: {}".format(x_test.shape))
print("y_train: {}".format(y_train.shape))
print("y_test: {}".format(y_test.shape))


# In[53]:


#standardization
scaler = StandardScaler()
scaler

