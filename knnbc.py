import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score 

data = pd.read_csv("https://www.sciencebuddies.org/ai/colab/breastcancer.csv?t=AQVmtI91jPplJSrKVHTjCpbAc31FOWHvnNSIQhAUIiWOqA")

data.drop('id', axis=1, inplace=True)

numerical_columns = ['radius_mean', 'texture_mean', 'perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave_points_mean','symmetry_mean',
                     'fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave_points_se','symmetry_se',
                     'fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst',
                     'concave_points_worst','symmetry_worst','fractal_dimension_worst']

data[numerical_columns] = (data[numerical_columns] - data[numerical_columns].min()) / (data[numerical_columns].max() - data[numerical_columns].min())

data['diagnosis'].unique()
label_encoder = LabelEncoder()

data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])

X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

precision = precision_score(y_test, y_pred)
print("Precision:", precision)

recall = recall_score(y_test, y_pred)
print("Recall:", recall)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X_pca, y)

h = 0.02
x_min, x_max = X_pca[:, 0].min() - 0.1, X_pca[:, 0].max() + 0.1
y_min, y_max = X_pca[:, 1].min() - 0.1, X_pca[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title(f'KNN Decision Boundary (k={k}) - PCA Reduced')
plt.show()

neighbors = np.arange(1, 21)

accuracy_scores = []
precision_scores = []
recall_scores = []

for n in neighbors:
    knn = KNeighborsClassifier(n_neighbors=n)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

    precision = precision_score(y_test, y_pred)
    precision_scores.append(precision)

    recall = recall_score(y_test, y_pred)
    recall_scores.append(recall)
    
plt.plot(neighbors, accuracy_scores, label='accuracy')
plt.plot(neighbors, precision_scores, label='precision')
plt.plot(neighbors, recall_scores, label='recall')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.title('Comparison of Performance Metrics for Different Numbers of Neighbors')
plt.legend()
plt.show()