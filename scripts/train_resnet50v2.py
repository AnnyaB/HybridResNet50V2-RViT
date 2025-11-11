"""
train_resnet50v2.py
Replicatin' the ResNet50V2 experiment reported in Rasool et al. (2024)
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 100
NUM_CLASSES = 4

train_dir = "data/processed/train"
val_dir   = "data/processed/val"
test_dir  = "data/processed/test"

# --- These are our Data generators ---
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)
val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical")
val_data   = val_gen.flow_from_directory(val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical")
test_data  = test_gen.flow_from_directory(test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False)

# --- Model ---
base_model = ResNet50V2(include_top=False, weights="imagenet", input_shape=(224,224,3))
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(train_data, validation_data=val_data, epochs=30)

# --- Fine-tuning ---
for layer in base_model.layers[-30:]:
    layer.trainable = True
model.compile(optimizer=Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
history_ft = model.fit(train_data, validation_data=val_data, epochs=70)

# --- Evaluatin' ---
model.evaluate(test_data)
model.save("models/resnet50v2_rasool2024.h5")
