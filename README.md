# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM :

To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required :

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm :

### Step 1 :

Import the necessary python packages using import statements.

### Step 2 :

Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

### Step 3 :

Split the dataset using train_test_split.

### Step 4 :

Calculate Y_Pred and accuracy.

### Step 5 :

Print all the outputs.

### Step 6 :

End the Program.

## Program :

### Program to implement the SVM For Spam Mail Detection.
### DEVELOPED BY : ABRIN NISHA A
### REG NO : 212222230005

```
import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')

data.head()

data.info()

data.isnull().sum()

x=data["EmailText"].values
y=data["Label"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extractiaon.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output :

### DATA.HEAD() :

![image](https://github.com/Abrinnisha6/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118889454/0790abe8-f9d2-47bb-aa6f-97ec8c19fc6a)

### DATA.INFO() :

![image](https://github.com/Abrinnisha6/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118889454/b1da61ea-3463-480c-8615-11c3b7653442)

### DATA.ISNULL().SUM() :

![image](https://github.com/Abrinnisha6/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118889454/feebc9f0-a882-4644-909b-4e11cd5986d3)

### Y_PRED :

![image](https://github.com/Abrinnisha6/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118889454/de2c3b83-dd7e-4f3b-9522-1389ed1aa114)

### ACCURACY :

![image](https://github.com/Abrinnisha6/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118889454/95c9f6db-2a57-44c5-ac81-d41a47190c82)


## Result :

Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
