import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("C:/Users/DELL/Desktop/Heart_Disease_Prediction.csv")
df["Heart Disease"] = [1 if i == "Presence" else 0 for i in df["Heart Disease"]]
x = df.drop(["Heart Disease"], axis = 1)
y = df["Heart Disease"].values
X_train, X_test, y_train, y_test = train_test_split(x,y,random_state=42,test_size=0.33)

scaler = StandardScaler() 
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)


pclass= Perceptron(max_iter=10, eta0=0.1, random_state=0)
pclass.fit(X_train_scaler,y_train.ravel())
y_pred = pclass.predict(X_test_scaler)
acc=accuracy_score(y_test,y_pred)*100
print(accuracy_score(y_test,y_pred)*100)