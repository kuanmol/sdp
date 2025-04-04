import numpy as np
import cv2
from matplotlib import pyplot as plt

def add_noise(image, sigma=50):
    """Adds stronger Gaussian noise to an image (handles uint8 format)."""
    row, col, ch = image.shape
    mean = 0

    gauss = np.random.normal(mean, sigma, (row, col, ch)).astype(np.int16)
    noisy = np.clip(image.astype(np.int16) + gauss, 0, 255).astype(np.uint8)  # Ensure valid range
    return noisy

def add_haze(image, haze_factor=0.7):
    """Adds stronger synthetic haze to an image."""
    haze = np.full_like(image, 255, dtype=np.uint8)  # White haze overlay
    hazy_image = cv2.addWeighted(image, (1 - haze_factor), haze, haze_factor, 0)
    return hazy_image

def apply_noise_and_haze(image):
    """Applies both stronger noise and haze to the image."""
    image = add_noise(image, sigma=50)  # Increased noise
    image = add_haze(image, haze_factor=0.7)  # Increased haze
    return image

# Load Image from Path
image_path = r"D:\sdp\test\img_1.png"
image = cv2.imread(image_path)  # Load image in BGR format

# Convert BGR to RGB (for correct color display in Matplotlib)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Apply Both Noise and Haze
processed_image = apply_noise_and_haze(image)

# Display Images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(processed_image)
plt.title("Processed Image (Increased Noise + Haze)")
plt.axis("off")

plt.show()

# Save the Processed Image
output_path = r"D:\sdp\test\processed_img2.png"
cv2.imwrite(output_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))

print(f"Processed image saved to {output_path}")
