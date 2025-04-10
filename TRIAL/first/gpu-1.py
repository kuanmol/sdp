import numpy as np
from keras.api.models import load_model
from keras_preprocessing import image

loaded_model = load_model('my_image_classifier_model.keras')


test_image = image.load_img('img.png', target_size=(128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = test_image / 255.0

result = loaded_model.predict(test_image)
class_index = np.argmax(result[0])
class_labels = {0: 'cat', 1: 'dog', 2: 'human'}
prediction = class_labels[class_index]
print(f"Prediction: {prediction}")

print("Prediction Probabilities:", result[0])
