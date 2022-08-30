import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
sns.set()
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score
def my_function():
    scaler = MinMaxScaler()
    scaler.fit(x[['Cholesterol']])
    x['Cholesterol'] = scaler.transform(x[['Cholesterol']])
    scaler.fit(x[['Max HR']])
    x['Max HR'] = scaler.transform(x[['Max HR']])
    data["Heart Disease"] = [1 if i == "Presence" else 0 for i in data["Heart Disease"]]
    kmeans = KMeans(2)
    kmeans.fit(x)
    identified_clusters = kmeans.fit_predict(x)
    new_center = kmeans.cluster_centers_
    data_with_clusters = x.copy()
    data_with_clusters['Clusters'] = identified_clusters 
    df1 = x[data_with_clusters.Clusters==0]
    df2 = x[data_with_clusters.Clusters==1]
    acc=accuracy_score(data_with_clusters['Clusters'],data["Heart Disease"])
    print('accuracy in the function',acc*100)
    plt.scatter(new_center[:,0],kmeans.cluster_centers_[:,1],color='green',marker='*',label='centroid')
    plt.scatter(df1.Cholesterol,df1['Max HR'],color='black')
    plt.scatter(df2.Cholesterol,df2['Max HR'],color='red')
    plt.xlabel('Cholesterol')
    plt.ylabel('Max HR')
    plt.show()
    plt.legend()
    return data_with_clusters['Clusters']
data = pd.read_csv("C:/Users/DELL/Desktop/Heart_Disease_Prediction.csv")
x = data.iloc[:,[4,7]]
data["Heart Disease"] = [1 if i == "Presence" else 0 for i in data["Heart Disease"]]
kmeans = KMeans(2)
kmeans.fit(x)
identified_clusters = kmeans.fit_predict(x)
center=kmeans.cluster_centers_
data_with_clusters = x.copy()
data_with_clusters['Clusters'] = identified_clusters 
df1 = x[data_with_clusters.Clusters==0]
df2 = x[data_with_clusters.Clusters==1]
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color='green',marker='*',label='centroid')
plt.scatter(df1.Cholesterol,df1['Max HR'],color='black')
plt.scatter(df2.Cholesterol,df2['Max HR'],color='red')
plt.xlabel('Cholesterol')
plt.ylabel('Max HR')
plt.show()
acc1=accuracy_score(data_with_clusters['Clusters'],data["Heart Disease"])
print("accuracy of first Cluster",acc1*100)
new_clusters =  my_function()
while not np.array_equal(new_clusters,data_with_clusters['Clusters']):
 data_with_clusters['Clusters']=new_clusters
 new_clusters =  my_function()
 

acc=accuracy_score(new_clusters,data["Heart Disease"])
print("final accuracy",acc*100)


 

