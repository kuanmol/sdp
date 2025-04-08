import cv2
import numpy as np
import tensorflow as tf
from PIL import ImageFile
from keras_preprocessing import image

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Load the trained model (replace with the path to your model)
cnn = tf.keras.models.load_model('dog_human_cat_classifier_vgg16.h5')

# Initialize the camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

while True:
    ret, frame = cap.read()

    # Show the camera feed
    cv2.imshow('Camera', frame)

    # Wait for the 'c' key to capture the image
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        # Process the captured image
        test_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        test_image_resized = cv2.resize(test_image, (150, 150))
        test_image_resized = image.img_to_array(test_image_resized)
        test_image_resized = np.expand_dims(test_image_resized, axis=0)
        test_image_resized = test_image_resized / 255.0  # Rescale the image

        # Predict the class of the captured image
        result = cnn.predict(test_image_resized)
        class_index = np.argmax(result[0])
        class_labels = {0: 'cat', 1: 'dog', 2: 'human'}

        # Display the prediction
        prediction = class_labels[class_index]
        print(f"Prediction: {prediction}")

        # Draw bounding box and label on the frame
        h, w, _ = frame.shape
        top_left = (50, 50)
        bottom_right = (w - 50, h - 50)

        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        label_position = (top_left[0], top_left[1] - 10)
        cv2.putText(frame, prediction, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Show the frame with bounding box and label
        cv2.imshow('Camera with Bounding Box', frame)

    # Exit the loop if 'q' is pressed
    if key == ord('q'):
        break

# Release the camera and close the window after the loop ends
cap.release()
cv2.destroyAllWindows()
