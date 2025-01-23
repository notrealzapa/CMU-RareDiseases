import torch
import os
import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

class ElasticDeformation:
    def __init__(self, alpha=125, sigma=20):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, image):
        shape = image.shape[1:]
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
        for index in range(image.shape[0]):
            image[index] = map_coordinates(image[index], indices, order=1).reshape(shape)
        return image

class NoiseInjection:
    """
    This class injects random Gaussian noise into the given input image.
    """

    def __init__(self, mean: float = 0.0, std: float = 0.25) -> None:
        """
        Constructor method
        :param mean: Mean of the Gaussian noise
        :param std: Standard deviation of the Gaussian noise
        """
        self.mean = mean
        self.std = std

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Method to inject Gaussian noise into an image.
        :param image: Input image as a NumPy array
        :return: Transformed image with added noise
        """
        # Convert image to a PyTorch tensor and float type
        image_tensor = torch.tensor(image, dtype=torch.float32)
        # Add Gaussian noise
        noise = self.mean + torch.randn_like(image_tensor) * self.std
        # Apply noise and clip values to valid range [0, 255]
        noisy_image = torch.clamp(image_tensor + noise, 0, 255)
        return noisy_image.numpy().astype(np.uint8)

# Augmentation pipeline
def augment_image(image_path, output_dir, num_augmentations=15):
    # Load image
    image = cv2.imread(image_path)
    h, w, c = image.shape
    image_tensor = torch.tensor(image.transpose(2, 0, 1))  # Convert to CHW format for torch

    # Define augmentations
    elastic_deform = ElasticDeformation(alpha=125, sigma=20)
    noise_inject = NoiseInjection(mean=0.0, std=0.1)

    for i in range(num_augmentations):
        augmented = elastic_deform(image_tensor.numpy())  # Elastic deformation
        augmented = noise_inject(augmented)  # Noise injection
        augmented = augmented.transpose(1, 2, 0).astype(np.uint8)  # Convert back to HWC format

        # Save augmented image
        output_path = os.path.join(output_dir, f"aug_{i}_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, augmented)

# Batch process images
input_dir = "/project/workspace/Rare/Sickle/positive"
output_dir = "/project/workspace/Rare/Sickle/AdvancedAugmented/"
os.makedirs(output_dir, exist_ok=True)

for image_file in os.listdir(input_dir):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        augment_image(os.path.join(input_dir, image_file), output_dir, num_augmentations=20)
