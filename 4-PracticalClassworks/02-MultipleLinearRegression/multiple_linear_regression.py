# Multipler Linear Regression formülü => y = b_0 + b_1*x_1 + b_2*x_2 + ... + b_n*x_n
# Bizim uyguluyacağımız Multiple Linear Regression formülü => maas = b_0 + b_1*deneyim + b_2*yas
# maas: dependent variable
# deneyim, yas = independent variable

# Gerekli kütüphaneleri import edelim
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Verisetini import edelim
df = pd.read_csv("dataset/multiple_linear_regression_dataset.csv", sep=";")

X = df.iloc[:, [0, 2]].values
y = df["maas"].values.reshape(-1, 1)

# Modelimizi oluşturalım
multipler_linear_regression = LinearRegression()
multipler_linear_regression.fit(X, y)

# Intercept değerine ulaşalım
b_0 = multipler_linear_regression.intercept_
print(f"b_0(intercept): {b_0}")  # 10376.62747228

# Coefficient değerlerine ulaşalım
coefficients = multipler_linear_regression.coef_
# [[1525.50072054 -416.72218625]]
print(f"b_1 and b_2(coefficients): {coefficients}")

# Predict işlemini gerçekleştirelim
# Deneyim : 10
# Yaş : 35
values = np.array([10, 35])
prediction = multipler_linear_regression.predict(np.array([values]))
print(prediction) # 11046.35815877

# Predict işlemini gerçekleştirelim
# Deneyim : 5
# Yaş : 35
values = np.array([5, 35])
prediction = multipler_linear_regression.predict(np.array([values]))
print(prediction) # 3418.85455609