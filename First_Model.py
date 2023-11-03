import streamlit as st1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


header=st1.container()

with header:
    st1.title('Welcome to world of machine learning')


url="https://github.com/sunilqp/Streamlitapp/blob/main/wine.csv"
winedf=pd.read_csv(url)
# Data Cleaning
winedf.info()
# Column Name has special Char to to change it 

winedf.columns=winedf.columns.str.replace("###",'')

winedf.info()
winedf.columns=winedf.columns.str.replace('*','')

winedf.info()
winedf.dtypes
winedf.duplicated()
winedf.info()
winedf.describe()
cat_col = [col for col in winedf.columns if winedf[col].dtype == 'object']
print('Categorical columns :',cat_col)
# Numerical columns
num_col = [col for col in winedf.columns if winedf[col].dtype != 'object']
print('Numerical columns :',num_col)
winedf.columns=winedf.columns.str.strip()
winedf.columns
winedf['Ash'].unique()
winedf.Ash=winedf.Ash.str.replace('@','')
winedf['Ash'].unique()
winedf.Ash=winedf.Ash.str.replace('','0')


winedf.Ash=winedf.Ash.astype(float)
winedf['phenols'].unique()

winedf.phenols=winedf.phenols.str.replace('#','0')
winedf['phenols'].unique()
winedf.phenols=winedf.phenols.astype(float)
winedf['Nonflavanoid_phenols'].unique()
winedf.Nonflavanoid_phenols=winedf.Nonflavanoid_phenols.str.replace('#','0')
winedf['Nonflavanoid_phenols'].unique()
winedf.Nonflavanoid_phenols=winedf.Nonflavanoid_phenols.astype(float)
winedf['customer_segment'].unique()
Y6=winedf['customer_segment']

winedf['customer_segment']=Y6

Y6=winedf['customer_segment']
winedf.customer_segment=winedf.customer_segment.str.upper()
# winedf.customer_segment=winedf.customer_segment.map({'1':1, 'ONE':1, 'one':1, 'OnE':1, '2':2, 'Two':2, 'TWO':2, '3':3, 'Three':3})

winedf.customer_segment = winedf.customer_segment.map({'ONE':1, '1':1, '2':2, 'TWO':2, '3':3, 'THREE': 3})
winedf.customer_segment.unique()


print(winedf['customer_segment'].unique())
winedf.dtypes
# Remove outlier
list1 = list(winedf.columns)

print("shape of dataframe before removal of Outlier ", winedf.shape)
for x in list1:
    mean = winedf[x].mean()
    sd = np.std(winedf[x])
    upper_limit = mean + 3*sd
    lower_limt = mean - 3*sd
    newwinedf = pd.DataFrame()
    newwinedf = winedf[(winedf[x] >= upper_limit) | (winedf[x] <= lower_limt)]
    
    if len(newwinedf)>0:
        print("Outlier is there in the column ", x)
        winedf = winedf[(winedf[x] < upper_limit) & (winedf[x] > lower_limt)]
        print("Removed outlier from column ", x)
    else:
        print("There is no outlier in the column ", x)
    print("----------------------------------------------------------")
    
print("shape of dataframe after removal of Outlier ", winedf.shape)

print(newwinedf.head())
winedf.head()
newwinedf.info()
newwinedf.head()
fig = plt.figure(figsize =(10, 7))
plt.boxplot(winedf)


plt.title('Box Plot')
plt.show()
# Check Heathmap

#sns.heatmap(winedf.corr(), annot = True, cbar = True)
#plt.show()
winedf.head(1)
# More check
print(winedf.isnull().sum())
print(winedf.isna().sum())

winedf.dropna()


X=winedf.drop(['customer_segment'],axis=1)
y=winedf['customer_segment']
#scale values of X and Y
sc = StandardScaler()
X= sc.fit_transform(X)
X_train, X_test,y_train,y_test = train_test_split(X,y,random_state=3 ,test_size =0.3)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print(y_train.unique())
df = y_train.replace([np.nan, -np.inf], 0)
y_train.dropna()
print(y_train.unique())
#Create Model
clf = LogisticRegression()
clf.fit(X_train,y_train)

predicate_value = clf.predict(X_test)
print(' Accuracy score is : ', accuracy_score(predicate_value,y_test))
print('Confusion Matrix is : ', confusion_matrix(predicate_value,y_test))

plt.plot(y_test,predicate_value)
plt.show()
sns.pairplot(winedf,
             corner=True)

plt.show()
