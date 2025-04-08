import cv2
import numpy as np


def add_haze(image, intensity=0.5):
    """Add synthetic haze to the image by blending with white."""
    haze = np.full_like(image, 255)  # white image
    hazed_img = cv2.addWeighted(image, 1 - intensity, haze, intensity, 0)
    return hazed_img


def add_noise(image, noise_type="gaussian", mean=0, var=20):
    """Add Gaussian noise to the image."""
    if noise_type == "gaussian":
        sigma = var ** 0.5
        gaussian = np.random.normal(mean, sigma, image.shape).astype(np.float32)
        noisy_img = image.astype(np.float32) + gaussian
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        return noisy_img
    else:
        raise NotImplementedError("Only 'gaussian' noise is implemented.")


# Load image
image_path = "D:/sdp/test/img_13.png"
image = cv2.imread(image_path)

# Apply haze and noise
hazed = add_haze(image, intensity=0.4)
hazed_noisy = add_noise(hazed, var=30)

# Save result
output_path = "output_hazed_noisy.jpg"
cv2.imwrite(output_path, hazed_noisy)

# Optionally display
# cv2.imshow("Hazed & Noisy", hazed_noisy)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
