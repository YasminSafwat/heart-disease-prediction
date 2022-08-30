import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


dataset = pd.read_csv('C:/Users/DELL/Desktop/Heart_Disease_Prediction.csv')
print(dataset.head())
print(dataset.info())
print(dataset.isnull().sum())
print(dataset.describe())
correlation = dataset.corr()
sns.heatmap(dataset.corr(),cmap="Blues")





X = dataset.drop(['Heart Disease'],axis=1)
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
