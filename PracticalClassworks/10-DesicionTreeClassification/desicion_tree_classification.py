# Gerekli kütüphaneleri import edelim
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Verisetimizi import edelim
data = pd.read_csv("dataset/data.csv")

# Modelimizi oluşturmak için gereksiz olan sütunları çıkaralım
data.drop(["id", "Unnamed: 32"], axis=1, inplace=True)


# Veri setindeki "diagnosis" feature'u bizim output feature'umuz olacak.
# Aynı zamanda bu feature bir sınıflandırma sütunudur.
# İçeriğindeki M ve B verilerini 1 ve 0 yaparak modelimizde kullanılmaya uygun hale getireceğiz.
data["diagnosis"] = [1 if label == "M" else 0 for label in data["diagnosis"]]

# x ve y eksenlerini belirleyelim
y = data.diagnosis.values
X_data = data.drop(["diagnosis"], axis=1)


# Normalizasyon gerçekleştirelim.
# Normalizasyon ile her bir feature'un model tarafından eşit değerlendirilmesini sağlıyoruz.
# Formül: (X - min(X)) / (max(X) - min(X))
X = (X_data - np.min(X_data))/(np.max(X_data)-np.min(X_data))


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42)


desicionTree = DecisionTreeClassifier()
desicionTree.fit(X_train, y_train)

print("Accuracy of Desicion Tree Classifier Model: ",
      desicionTree.score(X_test, y_test))
