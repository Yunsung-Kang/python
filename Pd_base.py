#packages load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
d= {"one" : range(20), "two" : np.random.randn(20)}
df= pd.DataFrame(d)

df.head()
df.tail(1)

df.shape	#data size
df.describe()	#data summary

df2=pd.DataFrame({'first':[2,2,2,2,4,4,6,6,6,6,6,6,10],
                  'second':['one','one','one','two','two','three','three','three','three','three','four','four','four']})
df2.value_counts()
df2.mode()

df.sort_values(by='two')	#오름차순
df.sort_values(by='two',ascending=False)	#내림차순
df['two'].sort_values()

# fillna() = null 값을 채워줌.

df3= pd.DataFrame({
    'unif' : np.random.uniform(-3,3,20),
    'norm' : np.random.normal(0,1,20)
})

# 시각화
plt.boxplot(df)
plt.show()

plt.plot(df)
plt.show()