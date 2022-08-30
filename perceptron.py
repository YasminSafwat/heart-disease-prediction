import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("C:/Users/DELL/Desktop/Heart_Disease_Prediction.csv")
x = df.drop(["Heart Disease"], axis = 1)
y = df["Heart Disease"].values
X_train, X_test, y_train, y_test = train_test_split(x,y,random_state=42,test_size=0.32)


mlp = MLPClassifier(
       max_iter=750,
       alpha=0.1,
       random_state=42
      )
mlp.fit(X_train, y_train)

mlp_predict = mlp.predict(X_test)
accuracy=accuracy_score(y_test, mlp_predict) * 100

print('MLP Accuracy: {:.2f}%'.format(accuracy))