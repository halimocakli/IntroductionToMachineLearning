# Öncelikle gerekli kütüphaneleri import edelim
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Üzerinde çalışacağımız veri setini import edelim
data = pd.read_csv("dataset/data.csv")

# Üzerinde çalıştığımız veri seti, bir tümörün iyi huylu ya da kötü huylu olduğunu belirten verilerden müteşekkildir.
# M : Malignant
# B : Benign
# Veri seti içerisindeki feature'ları inceleyelim
print(data.info())

# Veri seti içerisindeki "id" ve "Unnamed: 32" feature'ları modelimizin eğitimi için gereksiz.
# Bu sütunları veri setinden çıkaralım
data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)

# diagnosis içerisindeki classları ayıralım
M = data[data["diagnosis"] == "M"]
B = data[data["diagnosis"] == "B"]

# Veriyi görselleştirelim
plt.scatter(M.radius_mean, M.texture_mean,
            color="red", label="Malignant", alpha=0.5)
plt.scatter(B.radius_mean, B.texture_mean,
            color="green", label="Benign", alpha=0.5)
plt.xlabel("Radius Mean")
plt.ylabel("Texture Mean")
plt.legend()
plt.show()

# Veri setindeki "diagnosis" feature'u bizim output feature'umuz olacak.
# Aynı zamanda bu feature bir sınıflandırma sütunudur.
# İçeriğindeki M ve B verilerini 1 ve 0 yaparak modelimizde kullanılmaya uygun hale getireceğiz.
data["diagnosis"] = [1 if label == "M" else 0 for label in data["diagnosis"]]

# Yeni dataframe'i inceleyelim
print(data.info())

# x ve y eksenlerini belirleyelim
X_data = data.drop(["diagnosis"], axis=1)
y = data["diagnosis"]

# Normalizasyon gerçekleştirelim.
# Normalizasyon ile her bir feature'un model tarafından eşit değerlendirilmesini sağlıyoruz.
# Formül: (X - min(X)) / (max(X) - min(X))
X = (X_data - np.min(X_data)) / (np.max(X_data) - np.min(X_data)).values

# Elimizdeki veriyi Train - Test olacak şekilde ayıralım
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# SVM modelimizi oluşturalım
svm = SVC(random_state=1)
svm.fit(X_train, y_train)

print("Accuracy of SVM Model: ", svm.score(X_test, y_test))