import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load your original, unaltered image
image_path = 'image_406314327901_f054.png'
img = cv2.imread(r'C:\Users\safit\OneDrive\GitHub\3D-US-reconstraction-for-detecting-facial-fracture\Dataset\Patient1\IMG_frames\\' + image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, _ = img.shape

# 2. Normalized control points tracking the hyperechoic bone surface
# (Adjust these exact coefficients/points if needed for pixel-perfection)
normalized_points = np.array([
    [0.42, 0.60],  # Lower-left tip of the visible bone reflection
    [0.50, 0.49],  
    [0.60, 0.42],  
    [0.70, 0.38],  
    [0.78, 0.36]   # Right edge where the interface meets the shadow
])

# Convert normalized coordinates to actual image pixels
pts = (normalized_points * [w, h]).astype(np.int32)

# 3. Fit a 2nd-degree polynomial for a perfectly smooth curve matching the anatomy
coeffs = np.polyfit(pts[:, 0], pts[:, 1], deg=2)
poly_func = np.poly1d(coeffs)

# Generate dense x points across the curve span
x_curve = np.linspace(pts[0, 0], pts[-1, 0], 200)
y_curve = poly_func(x_curve)

# 4. Plot the curve directly onto the true pixel array
plt.figure(figsize=(10, 8), dpi=300)
plt.imshow(img_rgb)
plt.plot(x_curve, y_curve, color='red', linewidth=2, label='Segmented Bone Surface')
plt.axis('off')
plt.tight_layout()
plt.show()