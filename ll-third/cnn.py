import numpy as np
import tensorflow as tf
from PIL import ImageFile
from keras.api.applications import EfficientNetB0
from keras.api.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.api.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.api.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Define paths to your dataset
train_dir = r"C:\Users\anmol\OneDrive\Desktop\sdp\power\dataset\training"
test_dir = r"C:\Users\anmol\OneDrive\Desktop\sdp\power\dataset\test"

# Image dimensions and batch size
img_width, img_height = 224, 224  # EfficientNet works well with 224x224
batch_size = 32

# Data augmentation and preprocessing for training data
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values to [0, 1]
    rotation_range=40,  # Randomly rotate images
    width_shift_range=0.2,  # Randomly shift images horizontally
    height_shift_range=0.2,  # Randomly shift images vertically
    shear_range=0.2,  # Shear transformations
    zoom_range=0.2,  # Randomly zoom images
    horizontal_flip=True,  # Randomly flip images horizontally
    vertical_flip=True,  # Randomly flip images vertically
    fill_mode='nearest'  # Fill missing pixels with the nearest value
)

# Preprocessing for testing data (only rescaling)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),  # Resize images to 224x224
    batch_size=batch_size,
    class_mode='categorical'  # Use categorical labels for multi-class classification
)

# Load testing data
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),  # Resize images to 224x224
    batch_size=batch_size,
    class_mode='categorical'
)

# Compute class weights to handle any class imbalance
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

# Load pre-trained EfficientNetB0 model
base_model = EfficientNetB0(
    weights='imagenet',  # Use pre-trained weights from ImageNet
    include_top=False,  # Exclude the fully connected layers at the top
    input_shape=(img_width, img_height, 3)  # Input image shape
)

# Freeze the base model (pre-trained layers)
base_model.trainable = False

# Build the transfer learning model
model = Sequential([
    base_model,  # Add the pre-trained EfficientNetB0 model
    GlobalAveragePooling2D(),  # Global average pooling to reduce dimensions
    Dense(512, activation='relu'),  # Fully connected layer with 512 units
    Dropout(0.5),  # Dropout to prevent overfitting
    Dense(256, activation='relu'),  # Another fully connected layer with 256 units
    Dropout(0.5),  # Dropout to prevent overfitting
    Dense(3, activation='softmax')  # Output layer with 3 units (dog, human, cat)
])

# Compile the model
model.compile(
    optimizer='adam',  # Adam optimizer
    loss='categorical_crossentropy',  # Loss function for multi-class classification
    metrics=['accuracy']  # Track accuracy during training
)

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=5,  # Stop training if no improvement for 5 epochs
    restore_best_weights=True  # Restore the best model weights
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',  # Monitor validation loss
    factor=0.2,  # Reduce learning rate by a factor of 0.2
    patience=3,  # Wait for 3 epochs before reducing learning rate
    min_lr=0.0001  # Minimum learning rate
)


# Learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


lr_callback = LearningRateScheduler(lr_scheduler)

# Train the model
epochs = 5  # Number of epochs
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,  # Steps per epoch
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size,
    callbacks=[early_stopping, reduce_lr, lr_callback],  # Add callbacks
    class_weight=class_weights  # Use class weights
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc:.4f}')

# Save the model
model.save('dog_human_cat_classifier_efficientnet.h5')
print("Model saved as 'dog_human_cat_classifier_efficientnet.h5'")
