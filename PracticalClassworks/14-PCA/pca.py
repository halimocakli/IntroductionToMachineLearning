# Gerekli kütüphaneleri import edelim
# PCA için sklearn.decomposition kütüphanesini import ediyoruz
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import pandas as pd

# Verisetimizi import edelim
# Verisetimiz içerisinde numpy arraylar mevcut
iris = load_iris()

data = iris.data
feature_names = iris.feature_names

# Output değişkeni
y = iris.target

dataFrame = pd.DataFrame(data, columns=feature_names)

# Hedef değişkenlerimizi class adı altında dataFrame'e ekleyelim
dataFrame["class"] = y

# Input variable - features
X = data

# PCA variable oluşturalım
# n_components : elimdekli verisetini hangi boyuta indirgeyeceğim?
# whiten: veriseti normalize edilecek mi?
pca = PCA(n_components=(2), whiten=True)

# Modelimizi fit edelim
# Bizim y ile bir işimiz yok, yalnızca feature indirgemeye çalışıyoruz.
# 4 boyutu 2 boyuta düşürecek modeli oluşturduk
pca.fit(X)

# Oluşturduğumuz boyut düşürme modelini uygulayalım
X_pca = pca.transform(X)

# Oluşan componentleri inceleyelim
print("Varience Ratio:", pca.explained_variance_ratio_)

# Boyut düşürmemize rağmen verimizin varyansının % kaçına sahibiz?
print("Sum: ", sum(pca.explained_variance_ratio_))

# 2D Data Visualization

# Oluşturduğumuz iki principal component'i dataFrame'e ekleyelim
dataFrame["p1"] = X_pca[:, 0]
dataFrame["p2"] = X_pca[:, 1]

color = ["red", "green", "blue"]

plt.figure(figsize=(9,9))
for each in range(3):
    plt.scatter(dataFrame["p1"][dataFrame["class"] == each], dataFrame["p2"]
                [dataFrame["class"] == each], color=color[each], label=iris["target_names"][each])

plt.legend()
plt.xlabel("P1")
plt.ylabel("P2")
plt.show()
