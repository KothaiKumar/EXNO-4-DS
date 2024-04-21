# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```python
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![image](https://github.com/KothaiKumar/EXNO-4-DS/assets/121215739/931125ba-4a4b-40bb-80b8-784681494655)
```python

data.isnull().sum()
```
![image](https://github.com/KothaiKumar/EXNO-4-DS/assets/121215739/3817e4e0-d11b-4f72-8e99-f909952b99cb)
```python

missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/KothaiKumar/EXNO-4-DS/assets/121215739/ac966f43-cd8e-4b5e-ad1d-77a836e297e4)
```python

data2=data.dropna(axis=0)
data2
```
![image](https://github.com/KothaiKumar/EXNO-4-DS/assets/121215739/d7377280-f49a-46bc-b235-9a95cd97ddef)
```python
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/KothaiKumar/EXNO-4-DS/assets/121215739/cc482865-3b05-469f-abc7-ce937258e4b6)
```python
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/KothaiKumar/EXNO-4-DS/assets/121215739/7b5c8869-39a8-46bf-b53b-64f13b71cf66)
```python
data2
```
![image](https://github.com/KothaiKumar/EXNO-4-DS/assets/121215739/dbf2df0e-f491-4357-8f65-a72112b783be)
```python
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/KothaiKumar/EXNO-4-DS/assets/121215739/782a4d7c-e8f1-47a8-bd06-5217afe47980)
```python

columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/KothaiKumar/EXNO-4-DS/assets/121215739/7673e5b9-32c3-40d2-b7e6-25fe60ad2246)
```python
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/KothaiKumar/EXNO-4-DS/assets/121215739/fd571647-e885-4d35-ad66-649e18a7d2f8)
```python
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/KothaiKumar/EXNO-4-DS/assets/121215739/9c80e0d7-f223-4832-9cf1-fe31a4ab80ee)
```python

x=new_data[features].values
print(x)
```
![image](https://github.com/KothaiKumar/EXNO-4-DS/assets/121215739/8fd3af0f-56c3-43a1-8b2d-c4c295a8594c)
```python

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/KothaiKumar/EXNO-4-DS/assets/121215739/fbfd12ea-45b9-47c6-88b3-44f4a76f7e29)
```python

prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/KothaiKumar/EXNO-4-DS/assets/121215739/e5a57362-8a80-4484-8de7-f94419e79161)
```python

accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/KothaiKumar/EXNO-4-DS/assets/121215739/130c03dc-5ce7-49b4-bf27-71d0d3895850)
```python

print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![image](https://github.com/KothaiKumar/EXNO-4-DS/assets/121215739/5954d451-8e3c-4808-a435-d82eb3930b3d)
```python

data.shape
```
![image](https://github.com/KothaiKumar/EXNO-4-DS/assets/121215739/96ae511e-d472-4861-8922-31bec2161123)
```python

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/KothaiKumar/EXNO-4-DS/assets/121215739/f11eefaa-b1a1-4786-8e31-68b6cb3aef08)
```python

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/KothaiKumar/EXNO-4-DS/assets/121215739/d9310fd1-79e6-4306-9d3a-61cd457c98e3)
```python

tips.time.unique()
```
![image](https://github.com/KothaiKumar/EXNO-4-DS/assets/121215739/326a2946-3d9f-4e8b-8ef2-7ef8942b3994)
```python

contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/KothaiKumar/EXNO-4-DS/assets/121215739/64be4b72-faa2-49ef-9458-d59ac06b40ab)
```python

chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![image](https://github.com/KothaiKumar/EXNO-4-DS/assets/121215739/61a949f7-5426-445b-aff2-9246a8dea6c2)


# RESULT:
Thus, Feature selection and Feature scaling has been used on the given dataset.
