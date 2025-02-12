# Beyin Tümörü Sınıflandırma Modeli

Bu proje, beyin MRI görüntülerini kullanarak beyin tümörü sınıflandırması yapmak için Convolutional Neural Network (CNN) tabanlı bir model geliştirmektedir.

## Veri Seti
  Kaggle Linki : https://www.kaggle.com/datasets/sartajbhuvaji brain-tumor-classification-mri 
Veri seti dört farklı sınıf içermektedir:

- Glioma Tumor

- Meningioma Tumor

- No Tumor

- Pituitary Tumor

Eğitim ve test verileri, dataset/Training/ ve dataset/Testing/ klasörlerinde yer almaktadır.

## Kullanılan Teknolojiler

- Python

- TensorFlow & Keras

- OpenCV

- NumPy & Matplotlib

- Seaborn

## Kurulum

Projeyi çalıştırmak için aşağıdaki kütüphaneleri yükleyin:
'''bash
-pip install tensorflow numpy matplotlib seaborn opencv-python scikit-learn
'''
## Çalıştırma

Projeyi çalıştırmak için aşağıdaki adımları takip edin:

Veri setini dataset/ klasörüne yerleştirin.

training.py dosyasını çalıştırın:
'''bash
-python training.py
'''
## Model Mimarisi

- 3 adet Conv2D katmanı

- MaxPooling katmanları

- Fully Connected Dense katmanlar

- Dropout katmanı ile overfitting önleme

## Eğitim ve Sonuçlar

- Model, Adam optimizasyon algoritması ve categorical_crossentropy kaybı ile eğitildi.

- Model, test seti üzerinde doğruluk hesapladı ve performansı karmaşıklık matrisi ile analiz edildi.
