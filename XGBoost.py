# packages
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import sklearn.metrics as mt

# fetch dataset 
bank_marketing = fetch_ucirepo(id=222) 
  
# data (as pandas dataframes) 
X = bank_marketing.data.features 
y = bank_marketing.data.targets 
  
# metadata 
print(bank_marketing.metadata) 
  
# variable information 
print(bank_marketing.variables)

#Bank Marketing data
X.head()
X.describe()
X.info()
X.shape

mv = X.isna()
mv.sum()
X['contact'].value_counts()
X['contact'].isna().sum()
X['contact'] = X['contact'].fillna('No')
X['contact'].value_counts()

X['job'] = X['job'].fillna('No')
X['job'].value_counts()

X.info()
X1 = X.drop(['poutcome'],axis=1)
X1.info()

y.value_counts()

# Label
X1['month'].value_counts()

def month_to_numeric(month) :
    if (month == "jan"):
        return 1
    elif (month == "feb"):
        return 2
    elif (month == "mar"):
        return 3
    elif (month == "apr"):
        return 4
    elif (month == "may"):
        return 5
    elif (month == "jun"):
        return 6
    elif (month == "jul"):
        return 7
    elif (month == "aug"):
        return 8
    elif (month == "sep"):
        return 9
    elif (month == "oct"):
        return 10
    elif (month == "nov"):
        return 11
    elif (month == "dec"):
        return 12
        
X1['month'] = X1['month'].apply(month_to_numeric)
X1['month'].value_counts()

X1.le = LabelEncoder()

for column in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact']:
    X1[column] = X1.le.fit_transform(X1[column])

X1.info()
X1.head()

y.le = LabelEncoder()

y = pd.DataFrame(y.le.fit_transform(y))
y.head()
y.value_counts()

X1['job'].value_counts()
X1['marital'].value_counts()
X1['education'].value_counts()
X1['loan'].value_counts()
X1['contact'].value_counts()

X1.info()

# train , test split
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size= 0.3)

# XGBoost
xgb = XGBClassifier(n_estimators = 500, learning_rate = 0.2, max_depth = 4, random_state = 32)
xgb.fit(X_train, y_train)
xgb.score(X_train,y_train)

y_pred = xgb.predict(X_test)
mt.accuracy_score(y_pred, y_test)
