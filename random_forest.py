import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("C:/Users/DELL/Desktop/Heart_Disease_Prediction.csv")
x = df.drop(["Heart Disease"], axis = 1)
y = df["Heart Disease"].values
X_train, X_test, y_train, y_test = train_test_split(x,y,random_state=42,test_size=0.32)

score=0
rf_scores = []
estimators = [10,100,200, 500, 1000]
for i in estimators:
    rf_classifier = RandomForestClassifier(n_estimators = i, random_state = 1)
    rf_classifier.fit(X_train, y_train)
    rf_scores.append(rf_classifier.score(X_test, y_test))
    y_predict_test = rf_classifier.predict(X_test)
    acc=accuracy_score(y_test, y_predict_test) * 100
    if score < acc:
      score = acc
    print('Random Forest  Accuracy: {:.2f}%'.format(acc))
    
final_acc=score
print('Random Forest  final Accuracy: {:.2f}%'.format(final_acc)) 
