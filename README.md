MNIST ANN Image Processing & Classification

Bu proje, MNIST veri setindeki el yazısı rakamları Yapay Sinir Ağı (ANN) kullanarak sınıflandırmayı amaçlamaktadır.
Model eğitimi öncesinde görüntü işleme teknikleri uygulanarak özellik çıkarımı güçlendirilmiştir.

📌 Kullanılan Veri Seti: MNIST
| Özellik       | Açıklama           |
| ------------- | ------------------ |
| Sınıf Sayısı  | 10 (0–9 rakamları) |
| Görsel Boyutu | 28x28 piksel       |
| Görsel Türü   | Grayscale          |
| Eğitim Verisi | 60,000             |
| Test Verisi   | 10,000             |


Amaç:
Bu rakamları ANN ile doğru bir şekilde tanımak / sınıflandırmak ✔️

🧠 Uygulanan Görüntü İşleme Adımları
Her görüntüye aşağıdaki işlemler uygulanır:

1️⃣ Histogram Eşitleme → Kontrast geliştirme

2️⃣ Gaussian Blur → Gürültü azaltma

3️⃣ Canny Edge Detection → Kenar belirleme

4️⃣ Flatten → 28×28 → 784 boyutuna indirgeme

5️⃣ Normalize (0–255 → 0–1)


Bu işlemler preprocess_images() fonksiyonu ile gerçekleştirilmiştir.


🏗️ Yapay Sinir Ağı Mimarisi
| Katman | Tür             | Aktivasyon | Nöron |
| ------ | --------------- | ---------- | ----- |
| 1      | Dense + Dropout | ReLU       | 128   |
| 2      | Dense           | ReLU       | 64    |
| Çıkış  | Dense           | Softmax    | 10    |

Optimizer: Adam
Loss: Sparse Categorical Crossentropy
Epoch: 50
Batch Size: 32


📈 Eğitim Sonuçları

🔹 Eğitim & doğrulama başarı grafikleri matplotlib ile gösterilmektedir.

📦 Kullanılan Kütüphaneler

TensorFlow / Keras

NumPy

Matplotlib

OpenCV (cv2)


▶️ Nasıl Çalıştırılır?

pip install tensorflow opencv-python matplotlib numpy

python mnist_ann.py
