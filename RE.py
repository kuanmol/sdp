from keras_preprocessing.image import ImageDataGenerator
from keras.api.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.api.models import Model
from keras.api.optimizers import Adam
from PIL import ImageFile
from keras.api.applications import VGG16
import numpy as np
import tensorflow as tf

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Define new dataset path
train_dir = r"D:\sdp\redd\train"
test_dir = r"D:\sdp\redd\test"
# train_dir = r"/workspace/sdp/redd/train"
# test_dir = r"/workspace/sdp/redd/test"

import cv2
import random


def apply_random_degradation(image):
    degrade_options = [
        add_gaussian_blur,
        add_gaussian_noise,
        add_haze_effect,
        adjust_contrast_randomly,
        add_jpeg_artifacts
    ]

    num_degradations = random.randint(1, 2)
    selected = random.sample(degrade_options, num_degradations)

    for degrade in selected:
        image = degrade(image)

    return image


def add_gaussian_blur(image):
    """Mild Gaussian blur"""
    if random.random() < 0.3:
        kernel_size = random.choice([3, 5])
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return image


def add_gaussian_noise(image):
    """Mild Gaussian noise"""
    if random.random() < 0.3:
        row, col, ch = image.shape
        sigma = 0.01  # previously variable, now fixed low
        gauss = np.random.normal(0, sigma, (row, col, ch))
        noisy = image + gauss
        return np.clip(noisy, 0, 1)
    return image


def add_haze_effect(image):
    """Light haze"""
    if random.random() < 0.2:
        haze_factor = random.uniform(0.85, 0.95)  # previously 0.3–0.7
        haze_color = np.ones_like(image) * 0.9
        hazy = image * haze_factor + haze_color * (1 - haze_factor)
        return np.clip(hazy, 0, 1)
    return image


def adjust_contrast_randomly(image):
    """Subtle contrast adjustment"""
    if random.random() < 0.3:
        alpha = random.uniform(0.85, 1.15)
        return np.clip(alpha * image, 0, 1)
    return image


def add_jpeg_artifacts(image):
    """Mild JPEG compression artifacts"""
    if random.random() < 0.2:
        quality = random.randint(50, 80)  # narrowed quality range
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv2.imencode('.jpg', (image * 255).astype(np.uint8), encode_param)
        decimg = cv2.imdecode(encimg, 1)
        return decimg.astype(np.float32) / 255.0
    return image


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,  # Random rotation up to 40 degrees
    width_shift_range=0.2,  # Random horizontal shift
    height_shift_range=0.2,  # Random vertical shift
    shear_range=0.2,  # Shear transformations
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Random horizontal flip
    brightness_range=[0.5, 1.5],  # Random brightness adjustment
    fill_mode='nearest',  # How to fill points outside boundaries
    channel_shift_range=50.0,  # Random channel shifts (color changes)

    preprocessing_function=lambda x: apply_random_degradation(x)
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)


def create_resnet_transfer_model(input_shape=(150, 150, 3), num_classes=3):
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='swish')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='swish')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    return model


model = create_resnet_transfer_model()
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

import pickle
from keras.api.callbacks import EarlyStopping, ReduceLROnPlateau

# Define early stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

epochs = 50
history = model.fit(
    training_set,
    validation_data=test_set,
    epochs=epochs,
    callbacks=[early_stop, reduce_lr]
)

model.save("third_model_da.h5")
with open("history3_da.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("Model and training history saved successfully!")

model = tf.keras.models.load_model('third_model_da.h5')

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

for layer in model.layers:
    if 'block5' in layer.name:
        layer.trainable = True
    elif isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

epochs = 30
history_ft = model.fit(
    training_set,
    validation_data=test_set,
    epochs=epochs,
    callbacks=[early_stop, reduce_lr]
)

model.save("model3_ft.h5")
with open("history3_finetuned.pkl", "wb") as f:
    pickle.dump(history_ft.history, f)

print("✅ Fine-tuning complete. Model and history saved.")
