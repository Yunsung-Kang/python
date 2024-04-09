# packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# fetch dataset 
bank_marketing = fetch_ucirepo(id=222) 
  
# data (as pandas dataframes) 
X = bank_marketing.data.features 
y = bank_marketing.data.targets 
  
# metadata 
print(bank_marketing.metadata) 
  
# variable information 
print(bank_marketing.variables)

#str
X.shape
X.head()
y.value_counts()
X.value_counts()
X.describe()
X.info()

# Label
y.le = LabelEncoder()
y = pd.DataFrame(y.le.fit_transform(y))
y.head()
y.value_counts()

X.info()
X1 = X.iloc[:,[0,5,9,11,12,13,14]]

# train_test_split
train_input, test_input, train_target, test_target = train_test_split(X1, y, test_size= 0.3)
print(train_input)
print(train_target)
# Lg
lg = LogisticRegression()
lg.fit(train_input, train_target)
print(lg.score(train_input, train_target))
print(lg.score(test_input, test_target))
print(lg.coef_)

# Lg pred
lg_pred = pd.DataFrame(lg.predict(test_input))
print(lg.score(test_input,lg_pred))