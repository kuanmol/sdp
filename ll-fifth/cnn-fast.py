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

# Ensure TensorFlow uses GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPUs available:", gpus)
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Using CPU.")

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

training_set = train_datagen.flow_from_directory(
    r"/workspace/ll-first/dataset/training",
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
)

test_set = test_datagen.flow_from_directory(
    r"/workspace/ll-first/dataset/test",
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
)
print(training_set.class_indices)
print(test_set.class_indices)

# Compute class weights to handle class imbalance
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(training_set.classes),
    y=training_set.classes
)
class_weights = dict(enumerate(class_weights))
batch = next(iter(training_set))  # Fetch one batch
print(f"Batch Length: {len(batch)}")  # Should be 2
print(f"Labels Shape: {batch[1].shape}")  # Should be (batch_size, num_classes)
print(f"Images Shape: {batch[0].shape}")  # Should be (batch_size, 150, 150, 3)

# Transfer Learning with VGG16
strategy = tf.distribute.MirroredStrategy()  # Enables multi-GPU support

with strategy.scope():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    base_model.trainable = False  # Freeze the base model

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])

    # Compile the Model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,  # Reduce learning rate by a factor of 0.2
    patience=5,  # Wait for 5 epochs before reducing learning rate
    min_lr=0.00001  # Minimum learning rate
)

# Learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_callback = LearningRateScheduler(lr_scheduler)

# Train the Model (Initial Training)
history = model.fit(
    training_set,
    validation_data=test_set,
    epochs=2,
    callbacks=[early_stopping, reduce_lr, lr_callback],
    class_weight=class_weights  # Added class weights
)

# Evaluate the Model
test_loss, test_accuracy = model.evaluate(test_set)
print(f"Test Accuracy after initial training: {test_accuracy * 100:.2f}%")

# Fine-Tuning: Unfreeze the top layers of the base model
base_model.trainable = True
for layer in base_model.layers[:-10]:  # Unfreeze the last 10 layers
    layer.trainable = False

# Recompile the model with a lower learning rate
with strategy.scope():
    model.compile(
        optimizer=Adam(learning_rate=1e-5),  # Very low learning rate for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

# Train the model again (Fine-Tuning)
history_fine = model.fit(
    training_set,
    validation_data=test_set,
    epochs=1,
    callbacks=[early_stopping, reduce_lr, lr_callback],
    class_weight=class_weights
)

model.save('my_image_classifier_model_finetuned.keras')

test_image = image.load_img('img_1.png', target_size=(150, 150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = test_image / 255.0

result = model.predict(test_image)
class_index = np.argmax(result[0])
class_labels = {0: 'cat', 1: 'dog', 2: 'human'}
prediction = class_labels[class_index]
print(f"Prediction: {prediction}")
