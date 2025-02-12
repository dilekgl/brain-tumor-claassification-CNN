import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import seaborn as sns

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import Input

# Kaggle'daki veri seti yolu
data_dirs = {
    "train": "dataset/Training/",
    "test": "dataset/Testing"
}

# Etiketler (sınıf isimleri)
labels = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

# Veri yükleme fonksiyonu
def load_data(data_dir, labels, img_size=224):
    """MRI görüntülerini yükleyip işleyerek numpy array'e dönüştüren fonksiyon."""
    X, y = [], []
    for label in labels:
        path = os.path.join(data_dir, label)
        for img_name in os.listdir(path):
            try:
                img = cv2.imread(os.path.join(path, img_name))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_size, img_size))
                X.append(img)
                y.append(labels.index(label))
            except Exception as e:
                print(f"Error loading image {img_name}: {e}")
    return np.array(X), np.array(y)

# Veriyi yükleyelim
X_train, y_train = load_data(data_dirs["train"], labels)
X_test, y_test = load_data(data_dirs["test"], labels)

# Normalizasyon işlemi
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

num_classes = len(labels)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(
    rotation_range=20,  
    width_shift_range=0.2,  
    height_shift_range=0.2,  
    shear_range=0.2,  
    zoom_range=0.2,  
    horizontal_flip=True,  
    fill_mode="nearest"
)
datagen.fit(X_train)

fig, axes = plt.subplots(1, 5, figsize=(15, 5))  

for i in range(5):
    axes[i].imshow(X_train[i])
    axes[i].axis("off")
    axes[i].set_title(labels[np.argmax(y_train[i])])

plt.show()



model = Sequential([
    Input(shape=(224, 224, 3)),  # Burada input_shape yerine Input() kullandık.
    
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.summary()

model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

import tensorflow as tf

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    verbose=1
)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

conf_matrix = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=labels))

