# packages
import pandas as pd
import numpy as np
import seaborn as sns
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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

# heatmap
X1.corr()
corr_matrix = X1.corr()
sns.heatmap(corr_matrix, cmap='coolwarm', annot= True)
plt.show()
sns.heatmap(X1)
plt.show()

# train_test_split
train_input, test_input, train_target, test_target = train_test_split(X1, y, test_size= 0.3)
print(train_input)
print(train_target)

# sm lg
import statsmodels.api as sm
log= sm.Logit(train_target, sm.add_constant(train_input)).fit()
print(log.summary())

X2 = X1.iloc[:,[2,3,6,7,8,11,12,13,14]]
x_train, x_test, y_train, y_test = train_test_split(X2, y, test_size= 0.3)

log1 = sm.Logit(y_train, sm.add_constant(x_train)).fit()
print(log1.summary())
np.exp(log1.params)

# Lg
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(train_input, train_target)
print(lg.score(train_input, train_target))
print(lg.score(test_input, test_target))
print(lg.coef_)

# Lg pred
lg_pred = pd.DataFrame(lg.predict(test_input))
print(lg.score(test_input,lg_pred))

print(metrics.mean_squared_error(test_target, lg_pred))

# DT
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt 
from sklearn.tree import plot_tree
from sklearn import metrics

dt = DecisionTreeClassifier(max_depth=3)
dt.fit(train_input, train_target)

plt.figure(figsize=(10,7))
plot_tree(dt, filled = True)
plt.show()
dt.score(train_input,train_target)
dt.score(test_input, test_target)

dt_pred = dt.predict(test_input)
print(metrics.mean_squared_error(test_target, dt_pred))