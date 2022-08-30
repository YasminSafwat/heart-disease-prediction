import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
data = pd.read_csv("C:/Users/DELL/Desktop/Heart_Disease_Prediction.csv")
x = data.iloc[:,[4,7]]
wcss=[]
for k in range(1,10):
 kmeans = KMeans(k)
 kmeans.fit(x)
 wcss.append(kmeans.inertia_)

number_clusters = range(1,10)
plt.plot(number_clusters,wcss,'bx-')
plt.title('The Elbow title')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
