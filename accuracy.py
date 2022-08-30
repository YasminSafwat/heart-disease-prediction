import pandas as pd

data = pd.read_csv("C:/Users/DELL/Desktop/Heart_Disease_Prediction.csv")

data["Heart Disease"] = [1 if i == "Presence" else 0 for i in data["Heart Disease"]]
presence = data[data["Heart Disease"]==1] 
absence = data[data["Heart Disease"]==0] 

overall_accuracy = max(len(presence),len(absence)) / (data.shape[0] - 1)

compare = dict()
for column in data.columns:
    total_feature_accuracy = []
    s = set(data[column].values)
    for value in s:
        x = len(presence[presence[column] == value])
        y = len(absence[absence[column] == value])
        acc_v = max(x,y)/(x+y)
        total_feature_accuracy.append(acc_v)
    compare[column] = sum(total_feature_accuracy)
        
print(compare)
