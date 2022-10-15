# Öncelikle gerekli kütüphaneleri import edelim
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Üzerinde çalışacağımız veri setini import edelim
data = pd.read_csv("dataset/data.csv")

# Üzerinde çalıştığımız veri seti, bir tümörün iyi huylu ya da kötü huylu olduğunu belirten verilerden müteşekkildir.
# Veri seti içerisindeki feature'ları inceleyelim
print(data.info())

# Veri seti içerisindeki "id" ve "Unnamed: 32" feature'ları modelimizin eğitimi için gereksiz.
# Bu sütunları veri setinden çıkaralım
data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)

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

# Row ve Column yerlerini değiştirelim
X_train = X_train.T
X_test = X_test.T
y_train = y_train.T
y_test = y_test.T

# Oluşturduğumuz yapının shape bilgisini öğrenelim
print(f"X_train Shape: {X_train.shape}")
print(f"X_test Shape: {X_test.shape}")
print(f"y_train Shape: {y_train.shape}")
print(f"y_test Shape: {y_test.shape}")


# Parametrelere ilk değerleri atama ve Sigmoid Fonksiyonu
# dimention: feature sayısı, weight sayısı.  Bu veri seti için 30'dur.
def initializeWeightsAndBias(dimention):
    # 0.01'lerden oluşan (30, 1) boyutunda bir numpy array oluşturalım
    w = np.full((dimention, 1), 0.01)
    b = 0.0

    return w, b


# weights, bias = initializeWeightsAndBias(30)
# Sigmoid Fonksiyonu: f(x) = 1 / 1 + e^-(x)
def sigmoid(z):

    y_head = 1 / (1 + np.exp(-z))
    return y_head


def forwardBackwardPropagation(weights, bias, X_train, y_train):
    # Forward Propagation
    # X_train.shape[1] ifadesi, scaling gerçekleştirmek için kullanıldı

    z = np.dot(weights.T, X_train) + bias
    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head) - (1 - y_train)*np.log(1-y_head)
    cost = (np.sum(loss)) / X_train.shape[1]

    # Backward Propagation
    # Türev ile eğim buluyoruz
    derivative_weight = (
        np.dot(X_train, ((y_head - y_train).T))) / X_train.shape[1]

    derivative_bias = np.sum(y_head - y_train) / X_train.shape[1]

    gradients = {"derivative_weight": derivative_weight,
                 "derivative_bias": derivative_bias}

    return cost, gradients


# Öğrenme parametrelerini güncelleme fonksiyonu
def update(weights, bias, X_train, y_train, learning_rate, number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    # number_of_iteration değeri kadar forwarding ve backwarding yapılır
    for i in range(number_of_iterarion):
        # Forward Propagation ve Backward Propagation yaparak Gradient değerlerini ve Cost değerini bul
        cost, gradients = forwardBackwardPropagation(
            weights, bias, X_train, y_train)
        cost_list.append(cost)

        # Bu noktada ağırlıkları güncelleme işlemi yapıyoruz
        weights = weights - learning_rate * gradients["derivative_weight"]

        # Bu noktada ağırlıkları güncelleme işlemi yapıyoruz
        bias = bias - learning_rate * gradients["derivative_bias"]

        # Her 10 adımda bir Cost değerini görmek istiyoruz.
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print("Cost after iteration %i: %f" % (i, cost))

    # Weight ve Bias parametrelerini güncelleyelim
    parameters = {"weight": weights, "bias": bias}
    plt.plot(index, cost_list2)
    plt.xticks(index, rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()

    return parameters, gradients, cost_list


# Prediction işlemi gerçekleştiren fonksiyon
def predict(weights, bias, X_test):
    # X_test, Forward Propagation işlemi için girdi olarak kullanılıyor
    z = sigmoid(np.dot(weights.T, X_test) + bias)
    y_head = np.zeros((1, X_test.shape[1]))

    # Eğer z değeri 0.5'ten büyükse tahminimiz 1 olacak -> y_head = 1
    # Eğer z değeri 0.5'ten küçükse tahminimiz 0 olacak -> y_head = 0
    for i in range(z.shape[1]):
        if z[0, i] <= 0.5:
            y_head[0, i] = 0
        else:
            y_head[0, i] = 1

    return y_head


def logisticRegression(X_train, y_train, X_test, y_test, learning_rate,  num_iterations):
    # Başlangıç değerleri veriliyor
    dimension = X_train.shape[0]
    weights, bias = initializeWeightsAndBias(dimension)
    
    # Learning Rate değerini değiştirmeyin
    parameters, gradients, cost_list = update(
        weights, bias, X_train, y_train, learning_rate, num_iterations)

    y_prediction_test = predict(
        parameters["weight"], parameters["bias"], X_test)
    y_prediction_train = predict(
        parameters["weight"], parameters["bias"], X_train)

    # Train/Test hata değerini yazdıralım
    print("Train Accuracy: {} %".format(
        100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("Test Accuracy: {} %".format(
        100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))


logisticRegression(X_train, y_train, X_test, y_test,
                   learning_rate=0.01, num_iterations=300)
