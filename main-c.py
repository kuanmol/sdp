import numpy as np
import tensorflow as tf
import cv2
from keras_preprocessing import image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Load the trained model
cnn = tf.keras.models.load_model('my_model.h5')

# Start the camera
cap = cv2.VideoCapture(0)  # Use the first camera (usually the default laptop camera)

# Set camera frame width and height
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('Camera', frame)

    # Wait for user to press 'c' to capture an image for prediction
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):  # Capture image when 'c' is pressed
        # Preprocess the captured image
        test_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert color format
        test_image_resized = cv2.resize(test_image, (64, 64))  # Resize to match the input size
        test_image_resized = image.img_to_array(test_image_resized)
        test_image_resized = np.expand_dims(test_image_resized, axis=0)

        # Normalize the image to match training data
        test_image_resized = test_image_resized / 255.0  # Rescale pixel values to [0, 1]

        # Make a prediction
        result = cnn.predict(test_image_resized)
        class_index = np.argmax(result[0])

        # Class labels
        class_labels = {0: 'cat', 1: 'dog', 2: 'human'}

        # Display the prediction
        prediction = class_labels[class_index]
        print(f"Prediction: {prediction}")

        # Get bounding box coordinates (assuming you're detecting faces or objects)
        h, w, _ = frame.shape
        top_left = (50, 50)  # Example coordinates for bounding box start (top-left corner)
        bottom_right = (w - 50, h - 50)  # Example coordinates for bottom-right corner

        # Draw bounding box (you can adjust the coordinates as needed)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        # Add the class label to the bounding box
        label_position = (top_left[0], top_left[1] - 10)
        cv2.putText(frame, prediction, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame with the bounding box
        cv2.imshow('Camera with Bounding Box', frame)

        # Break the loop after prediction
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
