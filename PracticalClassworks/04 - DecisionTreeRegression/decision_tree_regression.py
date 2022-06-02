from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Veri setini import edelim
dataFrame = pd.read_csv(
    "dataset/decision_tree_regression_dataset.csv", sep=";", header=None)

# Maç seyreden insanların tribündeki seviyeleri
X = dataFrame.iloc[:, 0].values.reshape(-1, 1)
# Her seviyenin ücreti
y = dataFrame.iloc[:, 1].values.reshape(-1, 1)

# Decision Tree Regression
tree_reg = DecisionTreeRegressor()

# Ağaç modelimizi oluşturuyoruz
tree_reg.fit(X, y)

# 6. seyirci seviyesinin ücretini tahmin etmeye çalışalım
print(tree_reg.predict([[6]]))  # 40
# Görüldüğü üzere yapılan tahmin doğru

# Bütün X değerleri için y tahmininde bulunalım
y_head = tree_reg.predict(X)

# Modelimizi görselleştirelim
plt.scatter(X, y, color="red")
plt.plot(X, y_head, color="green")
plt.xlabel("Tribün Seviyesi")
plt.ylabel("Ücret")
plt.show()

# Seviye aralıklarını split olarak kabul ettiğimizde eğri gibi değil de düz bir grafik bekliyorduk
# Bu şekilde olmasının sebebi ise yalnızca istediğimiz değerleri tahmin ettirmemizdi
# Daha geniş bir aralıkta tahmin işlemi yapmalıyız.

X_new = np.arange(min(X), max(X), 0.01).reshape(-1, 1)
y_head = tree_reg.predict(X_new)

plt.scatter(X, y, color="red")
plt.plot(X_new, y_head, color="green")
plt.xlabel("Tribün Seviyesi")
plt.ylabel("Ücret")
plt.show()

# Şimdi gördüğünüz üzere Desicion Tree Regression modeline uygun bir grafik oluştu
# Belirli bir split noktasına gelene(trübün seviyesi) kadar ücret aynı kalıyor ve ardından tek seferde düşüyor.
