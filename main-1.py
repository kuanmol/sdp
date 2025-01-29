from keras.api.models import load_model
from keras_preprocessing import image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


# Load the saved model
cnn = load_model('my_model.h5')


test_image = image.load_img('img_1.png', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = test_image / 255.0  # Rescale the image

result = cnn.predict(test_image)
class_index = np.argmax(result[0])
class_labels = {0: 'cat', 1: 'dog', 2: 'human'}
prediction = class_labels[class_index]
print(prediction)

