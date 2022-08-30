import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
data = pd.read_csv("C:/Users/DELL/Desktop/Heart_Disease_Prediction.csv")
data["Heart Disease"] = [1 if i == "Presence" else 0 for i in data["Heart Disease"]]
print(data.dtypes)
print("number of elements =",data.size)
print("number rows =",data.shape[0])
print("number columns =",data.shape[1])
print(data.head())
print(data.info())
print(data.isnull().sum())
print(data.describe())
correlation = data.corr()
sns.heatmap(data.corr(),cmap="Blues")
plt.show()

#Heart Disease
zeros=0
ones=0
for i in data["Heart Disease"]:
  if i == 1 :
    ones=ones+1
  else:
    zeros=zeros+1
    
size=[ones,zeros]    
names= "Presence","Absence"
plt.pie(size)
plt.pie(size, labels=names, labeldistance=1.2,wedgeprops = { 'linewidth' : 3, 'edgecolor' : 'white'} );
plt.show()

#EKG 
pd.crosstab(data["EKG results"],data["Heart Disease"]).plot(kind='bar')
plt.title("Heart disease per EKG results")
plt.xlabel("EKG results")
plt.ylabel("Amount")
plt.legend(['No Disease','Disease'])
plt.show()

#Age and Max Heart Rate
plt.scatter(x= data[data['Heart Disease']==1]['Age'], y = data[data['Heart Disease']==1]['Max HR'])
plt.scatter(x= data[data['Heart Disease']==0]['Age'], y = data[data['Heart Disease']==0]['Max HR'])
plt.title("Heart Disease in function of Age and Max Heart Rate")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate")
plt.legend(["Disease", "No Disease"]);
plt.show()

#Chest pain type
pd.crosstab(data["Chest pain type"],data["Heart Disease"]).plot(kind='bar')
plt.title("Heart disease per Chest pain type")
plt.xlabel("Chest pain type")
plt.ylabel("Amount")
plt.legend(['No Disease','Disease'])

#Sex male=0  female=1
pd.crosstab(data["Sex"],data["Heart Disease"]).plot(kind='bar')
plt.title("Heart disease per Sex")
plt.xlabel("Sex")
plt.ylabel("Amount")
plt.legend(['No Disease','Disease'])
plt.show()

#fasting blood sugar over 120
pd.crosstab(data["FBS over 120"],data["Heart Disease"]).plot(kind='bar')
plt.title("Heart disease per fasting blood sugar over 120")
plt.xlabel("fasting blood sugar over 120")
plt.ylabel("Amount")
plt.legend(['No Disease','Disease'])
plt.show()

#Thallium:Exposure to high levels of thallium can result in  harmful heart effects 
pd.crosstab(data["Thallium"],data["Heart Disease"]).plot(kind='bar')
plt.title("The effect of thallium on the heart disease")
plt.xlabel("Thallium")
plt.ylabel("Amount")
plt.legend(['No Disease','Disease'])
plt.show()

#Number of vessels fluro
pd.crosstab(data["Number of vessels fluro"],data["Heart Disease"]).plot(kind='bar')
plt.title("Heart disease per Number of vessels fluro")
plt.xlabel("Number of vessels fluro")
plt.ylabel("Amount")
plt.legend(['No Disease','Disease'])
plt.show()

#Slope of ST:Slope of heart rate
pd.crosstab(data["Slope of ST"],data["Heart Disease"]).plot(kind='bar')
plt.title("Heart disease per Slope of ST")
plt.xlabel("Slope of ST")
plt.ylabel("Amount")
plt.legend(['No Disease','Disease'])
plt.show()

#Exercise angina 
pd.crosstab(data["Exercise angina"],data["Heart Disease"]).plot(kind='bar')
plt.title("Heart disease per Exercise angina")
plt.xlabel("Exercise angina")
plt.ylabel("Amount")
plt.legend(['No Disease','Disease'])
plt.show()

#Heart Disease in function of Cholesterol and blood pressure
plt.scatter(x= data[data['Heart Disease']==1]['Cholesterol'], y = data[data['Heart Disease']==1]['BP'])
plt.scatter(x= data[data['Heart Disease']==0]['Cholesterol'], y = data[data['Heart Disease']==0]['BP'])
plt.title("Heart Disease in function of Cholesterol and blood pressure")
plt.xlabel("Cholesterol")
plt.ylabel("blood pressure")
plt.legend(["Disease", "No Disease"]);
plt.show()


plt.scatter(x= data[data['Heart Disease']==1]['Cholesterol'], y = data[data['Heart Disease']==1]['Max HR'])
plt.scatter(x= data[data['Heart Disease']==0]['Cholesterol'], y = data[data['Heart Disease']==0]['Max HR'])
plt.title("Heart Disease in function of Cholesterol and Max HR")
plt.xlabel("Cholesterol")
plt.ylabel("Max HR")
plt.legend(["Disease", "No Disease"]);
plt.show()
