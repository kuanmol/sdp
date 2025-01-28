import os
import numpy as np
from keras.api.layers import Dense, Flatten, MaxPooling2D, Conv2D
from keras.api.layers import Dropout
from keras.api.models import Sequential

import tensorflow as tf
from keras.src.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator, load_img,img_to_array
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


train_dir = 'dataset/train'
validation_dir = 'dataset/valid'
test_dir = 'dataset/test'

# 2. Data Augmentation and Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=64,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=64,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=64,
    class_mode='binary'
)

# 3. Define the CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 4. Train the Model
history = model.fit(
    train_generator,
    epochs=10,  # Adjust based on your dataset
    validation_data=validation_generator
)

# 5. Evaluate the Model
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.2f}")

# 6. Predict for a Single Image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Resize image
    img_array = img_to_array(img)                      # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)      # Add batch dimension
    img_array = img_array / 255.0                      # Normalize
    return img_array

def predict_image(image_path, model):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:  # 0.5 threshold
        print(f"The image '{image_path}' is classified as: DOG")
    else:
        print(f"The image '{image_path}' is classified as: HUMAN")

# Example: Replace with the path to your test image
image_path = 'Afghan_hound_00116.jpg'
predict_image(image_path, model)

