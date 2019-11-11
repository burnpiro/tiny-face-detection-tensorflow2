import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import numpy as np

base = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False)
base.trainable = False

x = base.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
preds = tf.keras.layers.Dense(1, activation='softmax')(x)

model = tf.keras.models.Model(inputs=base.input, outputs=preds)

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate), loss='categorical_crossentropy')

for layer in model.layers:
    print(layer.name, layer.trainable)

model.summary()