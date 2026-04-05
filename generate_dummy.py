import cv2
import numpy as np
import os

os.makedirs('raw_images', exist_ok=True)

# Generate 5 'high risk' images (mostly red/brown, low vegetation)
for i in range(5):
    img = np.random.randint(50, 150, (300, 300, 3), dtype=np.uint8)
    # increase red and blue, lower green
    img[:,:,0] += 50  # B
    img[:,:,1] -= 30  # G
    img[:,:,2] += 50  # R
    cv2.imwrite(f'raw_images/high_risk_{i}.png', img)

# Generate 5 'low risk' images (mostly green, high vegetation)
for i in range(5):
    img = np.random.randint(50, 150, (300, 300, 3), dtype=np.uint8)
    img[:,:,0] -= 20  # B
    img[:,:,1] += 80  # G
    img[:,:,2] -= 20  # R
    cv2.imwrite(f'raw_images/low_risk_{i}.png', img)

# Generate 5 'medium risk' images (mixed)
for i in range(5):
    img = np.random.randint(80, 180, (300, 300, 3), dtype=np.uint8)
    cv2.imwrite(f'raw_images/medium_risk_{i}.png', img)

print("Generated dummy images in raw_images/")
