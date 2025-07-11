import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Config
IMG_SIZE = 224
DATASET_DIR = "./Training/"
categories = ["glioma", "meningioma", "pituitary", "notumor"]
label_dict = {cat: i for i, cat in enumerate(categories)}

# Load and Preprocess Data
data = []
labels = []

print("Loading images...")
for category in categories:
    path = os.path.join(DATASET_DIR, category)
    for img_name in os.listdir(path):
        try:
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            labels.append(label_dict[category])
            print(f"{len(data)} files")
        except:
            continue

data = np.array(data, dtype='float32') / 255.0
labels = to_categorical(labels, num_classes=4)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels)

y_train_labels = np.argmax(y_train, axis=1)

class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
class_weight_dict = dict(enumerate(class_weights))

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# Loading the Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Training the Model
class_weight_dict = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}

history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_test, y_test),
                    epochs=15, class_weight=class_weight_dict)

# Evaluating Model
print("\nEvaluating model...")
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print(confusion_matrix(y_true, y_pred_classes))
print(classification_report(y_true, y_pred_classes, target_names=categories))

# Saving Model
model.save("better_brain_tumor_classifier_2.h5")
print("Model saved as better_brain_tumor_classifier.h5")

