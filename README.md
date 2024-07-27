# ML-TASK-02
Classifying Emails as Spam or Not Spam

#CODE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/kaggle/input/email-spam-classification-dataset-csv/emails.csv')
data.head()

data.info()

data.drop(columns=['Email No.'] , inplace = True)

data.head()

data.isna().sum()

data.shape

x = data.iloc[: , 0:3000]
y = data.iloc[: , 3000]

(x.shape , y.shape)

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2 , random_state=40)

# Build Model
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train , y_train)
y_pred = model.predict(x_test)
(np.array(y_test),y_pred)

# Accuracy
from sklearn.metrics import accuracy_score
print("Accuracy : " , np.round(accuracy_score(y_test , y_pred),4)*100)

# Cross Validation
from sklearn.model_selection import cross_val_score
cross_validation = cross_val_score(model,x_train , y_train , cv=10)
cross_validation.mean()*100
