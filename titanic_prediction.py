import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv("train.csv")

df['Age'].fillna(df['Age'].median(),inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)
df['Fare'].fillna(df['Fare'].median(),inplace=True)

le=LabelEncoder()
df['Sex']=le.fit_transform(df['Sex'])
df['Embarked']=le.fit_transform(df['Embarked'])

features=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
X=df[features]
y=df['Survived']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

log=LogisticRegression(max_iter=200)
log.fit(X_train,y_train)

rf=RandomForestClassifier(n_estimators=200,random_state=42)
rf.fit(X_train,y_train)

pred_log=log.predict(X_test)
pred_rf=rf.predict(X_test)

def evaluate(y_true,y_pred,name):
    print(f"\n{name}:")
    print("Accuracy:",accuracy_score(y_true,y_pred))
    print("Precision:",precision_score(y_true,y_pred))
    print("Recall:",recall_score(y_true,y_pred))
    print("F1-Score:",f1_score(y_true,y_pred))

evaluate(y_test,pred_log,"Logistic Regression")
evaluate(y_test,pred_rf,"Random Forest")

while True:
    print("\n--- Titanic Survival Predictor ---")
    try:
        Pclass=int(input("Enter Pclass (1/2/3): "))
        Sex=input("Enter Sex (male/female): ")
        Age=float(input("Enter Age: "))
        SibSp=int(input("Enter number of siblings/spouses aboard: "))
        Parch=int(input("Enter number of parents/children aboard: "))
        Fare=float(input("Enter Fare: "))
        Embarked=input("Embarked (C/Q/S): ")
    except:
        print("Invalid input! Try again.")
        continue

    Sex=0 if Sex.lower()=="female" else 1
    Embarked=0 if Embarked.upper()=="C" else 1 if Embarked.upper()=="Q" else 2

    user_input=pd.DataFrame([[Pclass,Sex,Age,SibSp,Parch,Fare,Embarked]],columns=features)

    pred=rf.predict(user_input)[0]
    print("\nPrediction:", "Survived" if pred==1 else "Not Survived")

    ch=input("\nDo you want to predict again? (y/n): ")
    if ch.lower()!='y':break
