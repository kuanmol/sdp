import numpy as np
import tensorflow as tf
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# 1. Data Augmentation and Preprocessing
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

# 2. Load Training and Test Data
training_set = train_datagen.flow_from_directory('dataset/train', target_size=(64, 64),
                                                 batch_size=32, class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_set = test_datagen.flow_from_directory('dataset/test',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

# 3. Build the CNN Model
cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Train the Model
cnn.fit(x=training_set, validation_data=test_set, epochs=1)

# 5. Test the Model with a New Image
test_image = image.load_img('Abbas_Kiarostami_0001.jpg', target_size=(64, 64))  # Your test image
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# 6. Make a Prediction
result = cnn.predict(test_image)

# 7. Output the Prediction
training_set.class_indices  # This will give you the class labels, e.g., 'dog': 0, 'human': 1

if result[0][0] == 1:
    prediction = 'human'
else:
    prediction = 'dog'

print(f"The image is classified as: {prediction}")
