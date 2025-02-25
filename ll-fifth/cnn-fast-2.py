import numpy as np
import tensorflow as tf
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from keras.api.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.api.models import Sequential
from keras.api.applications import VGG16
from keras.api.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.api.optimizers import Adam
from sklearn.utils import class_weight
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

# Ensure TensorFlow detects and uses the GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPUs available:", gpus)
else:
    print("No GPU found. Using CPU.")

# WSL2 File Paths
train_dir = r"/workspace/sdp/ll-first/dataset/training"
test_dir = r"/workspace/sdp/ll-first/dataset/test"

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# Load Data
training_set = train_datagen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=32, class_mode='categorical'
)
test_set = test_datagen.flow_from_directory(
    test_dir, target_size=(150, 150), batch_size=32, class_mode='categorical'
)

# Debugging: Check if data is loaded correctly
for batch in training_set:
    images, labels = batch
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    break

# Compute class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(training_set.classes),
    y=training_set.classes
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# Enable Multi-GPU Strategy (if available)
strategy = tf.distribute.MirroredStrategy()
print(f"Using {strategy.num_replicas_in_sync} GPU(s)")

# Model Definition
with strategy.scope():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    base_model.trainable = False  # Freeze base model

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')  # Ensure correct number of classes
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

# Training the Model
history = model.fit(
    training_set,
    validation_data=test_set,
    epochs=2,
    class_weight=class_weights
)

# Evaluate Model
test_loss, test_accuracy = model.evaluate(test_set)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
