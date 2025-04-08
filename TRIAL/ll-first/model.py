import tensorflow as tf
from PIL import ImageFile
from keras.api.applications import VGG16
from keras.api.callbacks import EarlyStopping, ModelCheckpoint
from keras.api.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.api.models import Sequential
from keras_preprocessing.image import ImageDataGenerator

# Allow truncated images to be loaded
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Data Augmentation for Training Data
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,  # Increased rotation range
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],  # Wider brightness range
    fill_mode='nearest',
    validation_split=0.2  # 20% of data for validation
)

# Rescale Testing Data (No Augmentation)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Load Training Data
training_set = train_datagen.flow_from_directory(
    'dataset/training',  # Path to training data
    target_size=(128, 128),  # Increased image size
    batch_size=64,  # Increased batch size
    class_mode='categorical',
    subset='training'  # Specify training subset
)

# Load Validation Data
validation_set = train_datagen.flow_from_directory(
    'dataset/training',  # Same path as training data
    target_size=(128, 128),
    batch_size=64,
    class_mode='categorical',
    subset='validation'  # Specify validation subset
)

# Load Testing Data
test_set = test_datagen.flow_from_directory(
    'dataset/test',  # Path to test data
    target_size=(128, 128),
    batch_size=64,
    class_mode='categorical'
)

# Transfer Learning with VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Freeze the base model initially

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

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,  # Increased patience
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    'best_model.keras',  # Save the best model
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# Train the Model
history = model.fit(
    training_set,
    validation_data=validation_set,
    epochs=5,  # Increased epochs
    callbacks=[early_stopping, model_checkpoint]
)

# Evaluate the Model on Test Data
test_loss, test_accuracy = model.evaluate(test_set)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Fine-Tuning (Optional)
# Unfreeze some layers of the base model and train with a lower learning rate
base_model.trainable = True
for layer in base_model.layers[:15]:  # Freeze the first 15 layers
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Very low learning rate
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.save('my_image_classifier_model.keras')

from keras_preprocessing import image
import numpy as np


# Function to make a prediction on a single image
def predict_image(image_path, model, class_labels):
    # Load the image and resize it to the target size
    img = image.load_img(image_path, target_size=(128, 128))

    # Convert the image to a numpy array
    img_array = image.img_to_array(img)

    # Expand dimensions to match the model's input shape (batch size of 1)
    img_array = np.expand_dims(img_array, axis=0)

    # Rescale the image (same as during training)
    img_array = img_array / 255.0

    # Make a prediction
    predictions = model.predict(img_array)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions[0])

    # Map the index to the class label
    predicted_class_label = class_labels[predicted_class_index]

    # Get the confidence score (probability) for the predicted class
    confidence = np.max(predictions[0])

    return predicted_class_label, confidence


# Define the class labels (must match the order used during training)
class_labels = {0: 'cat', 1: 'dog', 2: 'human'}

# Path to the image you want to predict
image_path = 'img.png'  # Replace with the path to your image

# Make a prediction
predicted_class, confidence = predict_image(image_path, model, class_labels)
print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence * 100:.2f}%")

model.save('my_image_classifier_model.keras')
# Final Evaluation on Test Data
test_loss, test_accuracy = model.evaluate(test_set)
print(f"Test Accuracy after Fine-Tuning: {test_accuracy * 100:.2f}%")
