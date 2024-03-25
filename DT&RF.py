# packages
import seaborn as sns   # load iris data
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt 
from sklearn.tree import plot_tree

# iris data
iris= sns.load_dataset('iris')
iris.head()
iris.describe()
iris.info()
iris['species'].value_counts()

# split train/test data
data = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
target = iris['species']
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.3, random_state=1234)

# DT
dt = DecisionTreeClassifier(random_state=1234)
dt.fit(train_input, train_target)
print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))

# DT visual
plt.figure(figsize=(10,7))
plot_tree(dt, filled = True, feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
plt.show()

# max_depth : root 제외 노드
plt.figure(figsize=(10,7))
plot_tree(dt, max_depth = 1, filled = True, feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
plt.show()

#-----------------------------------------------------------------------------------------------------------------------------

#RF packages
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# input
x = iris.iloc[:,[0,1,2,3]]
print(x.head())
x.value_counts()

# target
y= iris['species']
y.le = LabelEncoder()
y= pd.DataFrame(y.le.fit_transform(y))
y.value_counts()

# RF
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3)
rf = RandomForestClassifier(n_estimators=100, max_features= 3)
rf.fit(x_train,y_train)

# confusion matrix
y_pred = rf.predict(x_test)
print(y_pred)
print(metrics.accuracy_score(y_test,y_pred))