import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
data=pd.read_csv("C:/Users/DELL/Desktop/Heart_Disease_Prediction.csv")
data["Heart Disease"] = [1 if i == "Presence" else 0 for i in data["Heart Disease"]]
x = data.drop(["Heart Disease"], axis = 1)
y = data["Heart Disease"].values
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3, random_state = 42)
k = range(1,50,2) 
testing_accuracy = []
training_accuracy = []
score = 0 
for i in k:
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(x_train,y_train)
    y_predict_train = knn.predict(x_train)
    training_accuracy.append(accuracy_score(y_train,y_predict_train)) 
    y_predict_test = knn.predict(x_test)
    acc_score=accuracy_score(y_test,y_predict_test)
    testing_accuracy.append(acc_score) 
    if score < acc_score:
     score = acc_score
     best_k=i
sns.lineplot(k, training_accuracy)
sns.scatterplot(k, training_accuracy)
sns.lineplot(k, testing_accuracy)
sns.scatterplot(k,testing_accuracy)

plt.legend(['training accuracy', 'testing accuracy'])
plt.show()
print('This is the best K for KNeighbors Classifier: ',best_k, '\nAccuracy score is: ', score*100)
