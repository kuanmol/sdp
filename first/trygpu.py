import numpy as np
import tensorflow as tf
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from keras.api.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.api.models import Sequential
from keras.api.applications import VGG16
from keras.api.callbacks import EarlyStopping
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set memory growth for GPUs to avoid memory allocation errors
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Check if GPU is available
print("GPUs available:", tf.config.list_physical_devices('GPU'))

# Data Augmentation and Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# Rescale Testing Data (No Augmentation)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Load Training Data
training_set = train_datagen.flow_from_directory(
    '/workspace/first/dataset/train',
    target_size=(128, 128),  # Increased image size
    batch_size=32,
    class_mode='categorical'
)

# Load Testing Data
test_set = test_datagen.flow_from_directory(
    '/workspace/first/dataset/test',
    target_size=(128, 128),  # Increased image size
    batch_size=32,
    class_mode='categorical'
)

# Transfer Learning with VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Freeze the base model

# Build the Model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # Output layer for 3 classes
])

# Compile the Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Early Stopping to Prevent Overfitting
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

# Train the Model
history = model.fit(
    training_set,
    validation_data=test_set,
    epochs=4,  # Increase epochs
    callbacks=[early_stopping]
)

# Evaluate the Model
test_loss, test_accuracy = model.evaluate(test_set)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Make a Prediction on a Single Image
test_image = image.load_img('../test/img.png', target_size=(128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = test_image / 255.0  # Rescale the image

result = model.predict(test_image)
class_index = np.argmax(result[0])
class_labels = {0: 'cat', 1: 'dog', 2: 'human'}
prediction = class_labels[class_index]
print(f"Prediction: {prediction}")

# Save the entire model to a file
model.save('my_image_classifier_model.keras')  # Recommended format for Keras models
