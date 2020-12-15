from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score,GridSearchCV,KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score,confusion_matrix
import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('../out.csv')  

#target variable
y = df.converted

#dropping the target variable
df.drop('converted',
  axis='columns', inplace=True)

#creating an encoder
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(df)
x = enc.transform(df).toarray()

#saving the encoder
Pkl_Filename = "encoder_test_lead_scoring.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(enc, file)


#modeling
kf=KFold(n_splits=5,shuffle=True,random_state=0)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=0)

rf=RandomForestClassifier(criterion='gini',max_depth= 8, min_samples_leaf=6,n_estimators=12,random_state=0)
rf_boost=AdaBoostClassifier(base_estimator=rf,n_estimators=20,random_state=0).fit(xtrain,ytrain)
y_predicted=rf_boost.predict(xtest)
acc=accuracy_score(ytest,y_predicted)
print('Model Trained with an accuracy of',acc)


#saving the model
Pkl_Filename_2 = "1_test_lead_scoring.pkl"  

with open(Pkl_Filename_2, 'wb') as file:  
    pickle.dump(rf_boost, file)