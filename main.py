import tensorflow as tf
from keras.api import layers, models
from keras_preprocessing.image import ImageDataGenerator


from sklearn.utils import class_weight
import numpy as np

# Assuming your training labels are 0 for humans and 1 for dogs
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

# Define the CNN model
def create_cnn_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    return model


# Compile the model
def compile_model(model):
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])


# Prepare the data
def prepare_data(train_dir, validation_dir, target_size, batch_size):
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    return train_generator, validation_generator


# Main function
def main():
    # Define directories
    train_dir = 'path/to/train_data'
    validation_dir = 'path/to/validation_data'

    # Define image dimensions and batch size
    img_width, img_height = 150, 150
    input_shape = (img_width, img_height, 3)
    batch_size = 32

    # Prepare data
    train_generator, validation_generator = prepare_data(train_dir, validation_dir, (img_width, img_height), batch_size)

    # Create and compile the model
    model = create_cnn_model(input_shape)
    compile_model(model)

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
    )

    # Save the model
    model.save('dog_vs_human_model.h5')


if __name__ == "__main__":
    main()
