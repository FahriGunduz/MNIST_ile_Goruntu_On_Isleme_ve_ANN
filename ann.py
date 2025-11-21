"""
MNIST veri seti:
    rakamlama 0-9 toplamında 10 sınıf var
    28x28 piksel boyutunda resimler var
    grayscale resimler
    60000 eğitim, 10000 test verisi
    amacımız: ann ile bu resimleri tanımlamak yada sınıflandırmak
Image processing:
    histogram eşitleme: kontrast iyileştirme
    gaussian blur: gürültü azaltma
    canny edge detection: kenar tespiti
ANN (Artificial Natural Network) ile MNIST veri setini sınıflandırma:

libraries:
    tensorflow: keras ile ANN modeli oluşturma ve eğitme
    matplotlib: görselleştirme
    cv2: opencv image processing
"""
#import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Dropout
from tensorflow.keras.optimizers import Adam

#load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print( "x_train shape: ", x_train.shape)
print( "y_train shape: ", y_train.shape)
"""
x_train shape:  (60000, 28, 28)
y_train shape:  (60000,)
"""
#Image preprocessing
img=x_train[5] #ilk resmi 

stages = {"original": img} #orjinal resmi stages sözlüğüne ekle

img_hist_eq = cv2.equalizeHist(img) #histogram eşitleme
stages["histogram eşitleme"] = img_hist_eq

img_blur = cv2.GaussianBlur(img_hist_eq, (5,5), 0) #gaussian blur
stages["gaussian blur"] = img_blur

img_canny = cv2.Canny(img_blur, 50, 150) #canny kenar eşitleme
stages["canny kenarları"] = img_canny

#görselleştirme
"""
fig , axes = plt.subplots(2,2, figsize=(6,6))
exes = axes.flat
for ax, (title, im) in zip(exes, stages.items()):
    ax.imshow(im, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.suptitle("Image Processing Stages")
plt.tight_layout()
plt.show()
"""
#preprocessing fonksiyonu
def preprocess_images(images):
    """
    Histogram eşitleme yapacak
    gaussian blur uygulayacak
    canny ile kenar tespiti yapacak
    flattering 28x28 -> 784 boyutuna değiştirecek
    normalizasyon yapacak 0-255 arasıda 0-1 arası çevirme
    """
    img_hist_eq = cv2.equalizeHist(images) #histogram eşitleme
    img_blur=cv2.GaussianBlur(img_hist_eq, (5,5), 0) #gaussian blur
    img_edges=cv2.Canny(img_blur, 50, 150) #canny kenar tespiti
    features = img_edges.flatten()/255.0 #flattening 28x28 -> 784
    return features

num_train = 60000
num_test = 10000

x_train= np.array([preprocess_images(img) for img in x_train[:num_train]])
x_train_sub = y_train[:num_train]

x_test= np.array([preprocess_images(img) for img in x_test[:num_test]])
x_test_sub = y_test[:num_test]

#ann model creation
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)), #ilk katman 128 nöron 28x28=784 giriş
    Dropout(0.5), # dropout katmanı, overfitting önleme için %50 dropout
    Dense(64, activation='relu'), #ikinci katman 64 nöron
    Dense(10, activation='softmax') #çıkış katmanı 10 nöron (0-9 rakamları için)
    
])

# compile model
model.compile(
    optimizer=Adam(learning_rate = 0.001), #optimizer
    loss= "sparse_categorical_crossentropy", #loss function
    metrics=["accuracy"] #değerlendirme metriği
)

print(model.summary())

#train model
history = model.fit( 
    x_train, x_train_sub, 
    validation_data=(x_test, x_test_sub),
    epochs=50, 
    batch_size=32, 
    verbose=2 
)

#evaluate model performance

test_loss, test_acc = model.evaluate(x_test, x_test_sub)
print(f"Test loss: , {test_loss:.4f},  Test accuracy: , {test_acc:.4f}")

#plot training history
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Kayıp Değerleri')
plt.xlabel('Epochs')
plt.ylabel('Kayıp')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Doğruluk Değerleri')
plt.xlabel('Epochs')
plt.ylabel('Doğruluk')
plt.legend()

plt.tight_layout()
plt.show()

