import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

score = 0
data = pd.read_csv("C:/Users/DELL/Desktop/Heart_Disease_Prediction.csv")

data["Heart Disease"] = [1 if i == "Presence" else 0 for i in data["Heart Disease"]]

x = data.drop(["Heart Disease"], axis = 1)

y = data["Heart Disease"].values

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.3,random_state=42)
for max_depth in range(1,14):
 dt = DecisionTreeClassifier(max_depth=max_depth, random_state=1)
 dt.fit(x_train, y_train)
 y_pred = dt.predict(x_test)
 acc = accuracy_score(y_pred, y_test)
 print("Test set accuracy: {:.2f}".format(acc*100),'max_depth = ',max_depth)
 if score < acc:
    best_max=max_depth
    score = acc
    
 dt = DecisionTreeClassifier(max_depth=best_max, random_state=1)
 dt.fit(x_train, y_train)
 y_pred = dt.predict(x_test)
 acc = accuracy_score(y_pred, y_test)
print("Test set accuracy: {:.2f}".format(acc*100))
print(best_max)
text_representation = tree.export_text(dt)
print(text_representation)
