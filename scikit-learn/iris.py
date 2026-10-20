import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

iris = load_iris() 

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['specied_name'] = df['target'].map(
    {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
)

df.to_csv('iris_dataset.csv', index=False)
X = df.drop(columns=['target', 'specied_name'])
y = df['specied_name']
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

model = DecisionTreeClassifier(max_depth=2, random_state=42, max_features=3)
model.fit(X_train_scale, y_train)

y_pred = model.predict(X_test_scale)

new_plant = [[5.1, 3.5, 1.4, 0.2]]
new_plant_scale = scaler.transform(new_plant)
#print(new_plant_scale)
prediction = model.predict(new_plant_scale)
prediction = le.inverse_transform(prediction)[0].upper()
print(prediction)

print('accuracy: ', accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
