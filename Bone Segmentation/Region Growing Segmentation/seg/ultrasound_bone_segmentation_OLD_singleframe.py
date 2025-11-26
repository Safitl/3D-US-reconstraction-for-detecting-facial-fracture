import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import flood, active_contour
from skimage.filters import gaussian
from skimage.draw import polygon
from skimage.morphology import opening, disk
from skimage.exposure import equalize_adapthist
from matplotlib.widgets import LassoSelector
from skimage.measure import find_contours
from skimage.filters import sobel

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img

def crop_ultrasound_region(img):
    # Approximate bounding box (manually tuned)
    return img[100:700, 200:800]

def preprocess_image(img):
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = equalize_adapthist(img / 255.0)  # normalize to [0, 1]
    clahe = (clahe * 255).astype(np.uint8)
    # Apply Gaussian blur
    return cv2.GaussianBlur(clahe, (7, 7), 0)

def region_growing(img, seed=(200, 160), tolerance=10):
    return flood(img, seed_point=seed, tolerance=tolerance)

def clean_mask(mask):
    return opening(mask.astype(np.uint8), disk(3))

def get_multiple_seeds_from_click(image, max_seeds=5):
    """
    Let the user click multiple seed points on the image.
    """
    seeds = []

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            y, x = int(event.ydata), int(event.xdata)
            print(f"Seed added: ({y}, {x})")
            seeds.append((y, x))
            if len(seeds) >= max_seeds:
                plt.close()

    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.set_title(f"Click up to {max_seeds} seed points (close window if done early)")
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    return seeds

def active_contour_refinement_from_mask(img, mask):
    """
    Extracts the largest contour from the mask and refines it with active contour.
    """
    smoothed = gaussian(img, sigma=1)
    
    # Extract the largest contour from region growing mask
    contours = find_contours(mask.astype(float), level=0.5)
    if not contours:
        raise ValueError("No contour found in the mask.")
    
    init = max(contours, key=len)  # choose the longest contour
    edges = sobel(smoothed)

    # Apply active contour model (snake)
    snake = active_contour(edges, init, alpha=0.0005, beta=2, gamma=0.01, w_line=0.5, w_edge=0.0)
    return init, snake

def create_mask_from_contour(img_shape, contour):
    mask = np.zeros(img_shape, dtype=np.uint8)
    rr, cc = polygon(contour[:, 0], contour[:, 1], img_shape)
    mask[rr, cc] = 255
    return mask

def main():
    #img_path = 'image_105551296540.png'  # Change this to your ultrasound image path
    img_path = r'C:\Users\safit\OneDrive\GitHub\3D-US-reconstraction-for-detecting-facial-fracture\Bone Segmentation\Region Growing Segmentation\Interactive-Region-Growing-Segmentation-master\image_105551296540.png'
    img = load_image(img_path)
    img = crop_ultrasound_region(img)
    blurred = preprocess_image(img)

    # Show blurred image
    plt.imshow(blurred, cmap='gray')
    plt.title("Preprocessed Image")
    plt.show()

    # Region growing
    # seed = (200, 160)  # adjust as needed
    seeds = [(178, 83),(194, 207),(218, 356),(167, 113),(208, 268)]
    #seeds = get_multiple_seeds_from_click(blurred, max_seeds=5)

    # Run region growing from each seed and combine
    combined_mask = np.zeros_like(blurred, dtype=bool)
    for seed in seeds:
        mask = region_growing(blurred, seed, tolerance=15)  # adjust tolerance
        combined_mask |= mask  # logical OR to combine all masks

    # Skip or lightly clean the result
    mask_clean = combined_mask.astype(np.uint8)

    plt.imshow(mask_clean, cmap='gray')
    plt.title("Region Growing (Cleaned)")
    plt.show()

    # Active contour refinement
    init, snake = active_contour_refinement_from_mask(blurred, mask_clean)

    plt.imshow(img, cmap='gray')
    plt.plot(init[:, 1], init[:, 0], '--r', label='Initial')
    plt.plot(snake[:, 1], snake[:, 0], '-b', label='Refined')
    plt.legend()
    plt.title("Active Contour")
    plt.show()

    # Create and save final mask
    final_mask = create_mask_from_contour(img.shape, snake)
    cv2.imwrite("segmentation_mask.png", final_mask)
    print("Saved: segmentation_mask.png")

if __name__ == "__main__":
    main()
