from keras.api.models import load_model
from keras_preprocessing.image import ImageDataGenerator

# Load the trained model
loaded_model = load_model('my_image_classifier_model.keras')

# Create an ImageDataGenerator for the test set
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Load test dataset from a directory
test_generator = test_datagen.flow_from_directory(
    'dataset/test',  # Replace with the actual path
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate the model
loss, accuracy = loaded_model.evaluate(test_generator)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Loss: {loss:.4f}")

import numpy as np
from sklearn.metrics import classification_report

# Get true labels
y_true = test_generator.classes  # True class labels

y_pred_probs = loaded_model.predict(test_generator)  # Predicted probabilities
y_pred = np.argmax(y_pred_probs, axis=1)  # Convert probabilities to class indices

class_labels = list(test_generator.class_indices.keys())

print(classification_report(y_true, y_pred, target_names=class_labels))
