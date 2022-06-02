from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

# Verisetini import edelim
df = pd.read_csv("dataset/polynomial_linear_regression.csv", sep=";")

# Modelimizin X ve y eksenlerini oluşturalım
X = df["araba_fiyat"].values.reshape(-1, 1)
y = df["araba_max_hiz"].values.reshape(-1, 1)

# Hatırlatma: values->Series tipindeki yapıyı array'e dönüştürür.
# Hatırlatma: reshape() metodu array'i sklearn kütüphanesini kullanabileceğimiz yapıya dönüştürür.

# Elde ettiğimiz verileri görselleştirelim
plt.scatter(X, y)
plt.xlabel("Araba Fiyatı")
plt.ylabel("Max Hız")

# Linear Regression deneyelim

# Linear Regression modelini oluşturalım
lr = LinearRegression()
lr.fit(X, y)

# X array'i içerisindeki verileri kullanarak predict işlemi yapalım
# Yani arabaların fiyatına göre göre max hız tahmininde bulunuyoruz
y_head = lr.predict(X)

# Prediction sonucunu görselleştirelim ve regresyon doğrusu üretelim
plt.plot(X, y_head, color="red", label="linear")

# 10000+ değerindeki bir aracın maksimum hızını tahmin edelim
print(f"Max speed of 10000$ car is {lr.predict([[10000]])}")

# Görüldüğü üzere 871 gibi oldukça büyük bir sonuç çıktı. Çünkü verisetimiz lineer regresyon ile modellenmeye uygun değil.
# Bu veriyi Polynomial Regression ile modelleyeceğiz.

# Linear Regression : y = b_0 + b_1 * x
# Multiple Linear Regression : y = b_0 + b_1 * x_1 + ... + b_n * x_n
# Polynomial Linear Regression : y = b_0 + b_1 * x^2 + ... + b_n * x^n

# Dereceyi 4 olacak şekilde belirliyoruz
polynomial_linear_regression = PolynomialFeatures(degree=4)
X_polinomial = polynomial_linear_regression.fit_transform(X)

# fit.transform() metodu ile oluşturduğumuz 4. dereceden denklemi lineer regresyon modeline veriyoruz
lr2 = LinearRegression()
lr2.fit(X_polinomial, y)

# Şimdi tahmin yapalım
y_head2 = lr2.predict(X_polinomial)

plt.plot(X, y_head2, color="green", label="polynomial")
plt.legend()
plt.show()
