import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
data=pd.read_excel("C:\Datasheets\pacific.xlsx")
#print(data)
data.Status=pd.Categorical(data.Status)
data['Status']=data.Status.cat.codes
print(data.Status)
sns.countplot(data['Status'],label='count')
plt.show()
#droooping data that are not used in making decesion it is purely our decesion
pred_columns=data[:]
pred_columns.drop(['Status'],axis=1,inplace=True)
pred_columns.drop(['Event'],axis=1,inplace=True)
pred_columns.drop(['Latitude'],axis=1,inplace=True)
pred_columns.drop(['Name'],axis=1,inplace=True)
pred_columns.drop(['ID'],axis=1,inplace=True)
prediction_var=pred_columns.columns
print(list(prediction_var))
#train and test
train,test=train_test_split(data,test_size=0.3)
print(train.shape)
print(test.shape)
X=train[prediction_var]
Y=train['Status']
print(list(train.columns))
test_X= test[prediction_var]
test_Y=test['Status']
print(list(test.columns))
#creaitng the decesion tree based on train data
model=tree.DecisionTreeClassifier()
model.fit(X,Y)
prediction=model.predict(test_X)
df=pd.DataFrame(prediction,test_Y)
print(df)
print(metrics.accuracy_score(prediction,test_Y))