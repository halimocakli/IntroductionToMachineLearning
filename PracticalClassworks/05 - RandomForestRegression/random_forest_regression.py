from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Veri setini import ediyoruz
# Tribün seviyeleri ve bu seviyelere göre fiyat bilgileri
dataFrame = pd.read_csv(
    "dataset/random_forest_regression_dataset.csv", sep=";", header=None)

# Tribün seviyeleri
X = dataFrame.iloc[:, 0].values.reshape(-1, 1)

# Tribün seviyelerine göe bilet fiyatları
y = dataFrame.iloc[:, 1].values.reshape(-1, 1)

# Random Forest Regression
# n_estimators : ağaç sayısı
# random_state : random durumunu sabitleyen id
randomForest = RandomForestRegressor(n_estimators=100, random_state=42)

# Modelimizi fit() edelim
randomForest.fit(X, y.ravel())

# Modelimizi kullanarak tahmin yapalım
print("7.5 Seviyesindeki fiyat:", randomForest.predict([[7.8]]))

# Modelimizin sonucunu görselleştirebilmek için tahmin değerleri oluşturalım
X_new = np.arange(min(X), max(X), 0.01).reshape(-1, 1)

# Tahmin değerlerini kullanarak tahmin sonuçlarına bakalım
y_head = randomForest.predict(X_new)

# Görselleştirelim
plt.scatter(X, y, color="red")
plt.xlabel("Tribün Level")
plt.ylabel("Ücret")
plt.show()

# Görselleştirelim
plt.plot(X_new, y_head, color="green")
plt.xlabel("Tribün Level")
plt.ylabel("Ücret")
plt.show()

# Görselleştirelim
plt.scatter(X, y, color="red")
plt.plot(X_new, y_head, color="green")
plt.xlabel("Tribün Level")
plt.ylabel("Ücret")
plt.show()
